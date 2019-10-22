# encoding:UTF-8
# @auth Wang
# @Date 2019/10/1
import tushare as ts
from time import sleep
from datetime import datetime
from queue import Queue, Empty
from threading import Thread
from multiprocessing.dummy import Pool
from ...trader.gateway import BaseGateway
from ...trader.constant import Direction
from ...trader.object import \
    (TickData, SubscribeRequest, OrderRequest, CancelRequest,
     LogData, Exchange, Product, ContractData)
from pandas import DataFrame
from typing import List, Dict


class ChinaStocksGateway(BaseGateway):
    # work day trade time
    open_trade_time: tuple = (9, 25,)
    close_trade_time: tuple = (15, 30,)
    column_name_map_dict = {}
    exchanges = [Exchange.SH, Exchange.SZ, Exchange.JJ]

    def __init__(self, event_engine):
        super(ChinaStocksGateway, self).__init__(event_engine, "ChinaStock")
        self.subscribe_stocks: list[str] = list()
        self.subscribe_funds: list[str] = list()
        self.all_stock_data: DataFrame = DataFrame()
        self.all_fund_data: DataFrame = DataFrame()
        self.bar_min: int = 1
        # related contract data
        self.query_contract: bool = False
        self.code_to_contract_dict: Dict = {}
        # related tick data
        self.tick_queue: Queue = Queue()
        self.thread: Thread = Thread(target=self._run)
        self.request_all_stocks_data: Thread = Thread(target=self._request_stocks)
        # control connect
        self.start_connect: bool = False
        self.today: datetime = datetime.now()
        # multiprocessing control
        self._tasks_tick_request_pool: Pool = Pool()
        self._request_task_dict: Dict = {}  # code_name map to request task

    def connect(self, setting: dict) -> None:
        """not necessary setting"""
        self.today = datetime.now()
        hour, minute = self.today.hour, self.today.minute
        # raise error if now time is not  trade time
        # if hour < self.open_trade_time[0] or hour > self.close_trade_time[0]:
        #     self.write_log(f"不在交易时间内,连接出错.")
        #     return
        # elif (hour == self.open_trade_time[0]
        #       and minute < self.open_trade_time[1]) or \
        #         (hour == self.close_trade_time[0] and minute > self.close_trade_time[1]):
        #     self.write_log(f"不在交易时间内,连接出错")
        #     return
        # start connect
        if not self.start_connect:
            self.start_connect = True
            self.request_all_stocks_data.start()
            self.thread.start()
        self.write_log(f"中国股票市场连接成功")

    def subscribe(self, req: SubscribeRequest) -> None:
        """simpleness subscribe stock"""

        stock_code = req.symbol
        exchange = req.exchange

        if stock_code in self.subscribe_stocks or stock_code in self.subscribe_funds:
            self.write_log(f"{stock_code}股票已经订阅。")
            return
        if exchange == Exchange.SZ or exchange == Exchange.SH:
            self.subscribe_stocks.append(stock_code)
        else:
            self.subscribe_funds.append(stock_code)

    def _request_stocks(self):
        """filter assign code of stock data that put it queue of tick """
        while True:
            try:
                # All stock and fund data requested by the Tushare api
                self.all_stock_data = ts.get_day_all()
                self.all_fund_data = ts.get_nav_open()
            except Exception:
                sleep(self.bar_min * 60)
                continue

            else:
                self.on_query_contracts()
                break
        # request  tick data
        while self.start_connect:
            try:
                self._request_all_tick_data()
            except Exception as e:

                sleep(self.bar_min * 60)
                continue
            else:
                sleep(self.bar_min * 60)

    def _request_all_tick_data(self) -> None:
        """multi threading use code which extracted  from the list then request data"""
        for code in self.subscribe_stocks:
            task_request = self._request_task_dict.get(code)
            # Only the requested is successful then it can be request data again
            if not task_request or task_request.successful():
                request_task = self._tasks_tick_request_pool.apply_async(
                    self.request_code_tick_data, args=(code,),
                    callback=self.call_back_request_tick_data)
                self._request_task_dict[code] = request_task
        for fund_code in self.subscribe_funds:
            task_request = self._request_task_dict.get(fund_code)
            if not task_request or task_request.successful():
                request_task = self._tasks_tick_request_pool.apply_async(self.request_code_tick_data,
                                                                         args=(fund_code,),
                                                                         callback=self.call_back_request_fund_data)
                self._request_task_dict[fund_code] = request_task

    def request_code_tick_data(self, code: str) -> (str, DataFrame()):
        while True:
            try:
                code_ticks_data: DataFrame() = ts.get_today_ticks(code, pause=0.3)
                if not code_ticks_data.empty:  # if the data frame is empty,it not to call back function
                    return code, code_ticks_data
            except Exception:
                sleep(self.bar_min * 60)
                continue

    def call_back_request_tick_data(self, code_data) -> None:
        """call back filter tick data if successful requested data"""
        code = code_data[0]
        ticks_data = code_data[1]
        if code not in self.code_to_contract_dict:
            # There is cache contract data locally
            contract = self.all_stock_data[self.all_stock_data.code == code]
            self.code_to_contract_dict[code] = contract
        contract: DataFrame = self.code_to_contract_dict[code]
        symbol: str = code
        name: str = contract.name.values[0]
        exchange = Exchange.SH if symbol.startswith("6") else Exchange.SZ
        # zero o index is newest data
        last_trade_time = datetime.strptime(ticks_data.time[0], "%H:%M:%S")
        # use today's year number and month number and day number.
        date_time = self.today.replace(hour=last_trade_time.hour,
                                       minute=last_trade_time.minute,
                                       second=last_trade_time.second)
        # set everyday trade price
        open_price = contract.open.values[0]
        high_price = contract.high.values[0]
        low_price = contract.low.values[0]
        volume = ticks_data.volume.sum()
        limit_up: float = contract.preprice.values[0] * 1.1
        limit_down: float = contract.preprice.values[0] * 0.9
        # new second of time trade data
        last_price = ticks_data.price[0]
        last_volume = ticks_data.volume[0]
        # Judge the order direction
        direction_type: str = ticks_data.type[0]
        if direction_type == "买盘":
            bid_price_1: float = last_price
            bid_volume_1: float = last_volume
            self.put_tick(symbol=symbol,
                          exchange=exchange,
                          datetime=date_time,
                          name=name,
                          volume=volume,
                          open_price=open_price,
                          high_price=high_price,
                          low_price=low_price,
                          limit_up=limit_up,
                          limit_down=limit_down, last_price=last_price,
                          last_volume=last_volume,
                          ask_price_1=bid_price_1,
                          ask_volume_1=bid_volume_1,
                          gateway_name=self.gateway_name)
        elif direction_type == "卖盘":
            ask_price_1: float = last_price
            ask_volume_1: float = last_volume
            self.put_tick(symbol=symbol,
                          exchange=exchange,
                          datetime=date_time,
                          name=name,
                          volume=volume,
                          open_price=open_price,
                          high_price=high_price,
                          low_price=low_price,
                          limit_up=limit_up,
                          limit_down=limit_down, last_price=last_price,
                          last_volume=last_volume,
                          ask_price_1=ask_price_1,
                          ask_volume_1=ask_volume_1,
                          gateway_name=self.gateway_name)
        # instance tick object
        else:
            self.put_tick(symbol=symbol,
                          exchange=exchange,
                          datetime=date_time,
                          name=name,
                          volume=volume,
                          open_price=open_price,
                          high_price=high_price,
                          low_price=low_price,
                          limit_up=limit_up,
                          limit_down=limit_down, last_price=last_price,
                          last_volume=last_volume,
                          gateway_name=self.gateway_name)

    def call_back_request_fund_data(self, code_data) -> None:
        code = code_data[0]
        fund_data = code_data[1]

        if code not in self.code_to_contract_dict:
            contract = self.all_fund_data[self.all_fund_data.symbol == int(code)]
            self.code_to_contract_dict[code] = contract
        contract = self.code_to_contract_dict.get(code)
        symbol = code
        name = contract.sname.values[0]
        open_price = contract.yesterday_nav.values[0]
        high_price = max(fund_data.price)
        low_price = min(fund_data.price)
        exchange = Exchange.JJ
        last_price = fund_data.price[0]
        last_volume = fund_data.volume[0]
        volume = fund_data.volume.sum()
        gateway_name = self.gateway_name
        last_trade_time = datetime.strptime(fund_data.time[0], "%H:%M:%S")
        date_time = self.today.replace(hour=last_trade_time.hour,
                                       minute=last_trade_time.minute,
                                       second=last_trade_time.second)
        trade_type = fund_data.type[0]
        if trade_type == "买盘":
            bid_price_1 = last_price
            bid_volume_1 = last_volume
            self.put_tick(symbol=symbol, exchange=exchange,
                          name=name, volume=volume, last_price=last_price,
                          datetime=date_time, gateway_name=gateway_name, bid_price_1=bid_price_1,
                          bid_volume_1=bid_volume_1, high_price=high_price, low_price=low_price)
            return
        elif trade_type == "卖盘":
            ask_price_1 = last_price
            ask_volume_1 = last_volume
            self.put_tick(symbol=symbol, exchange=exchange,
                          name=name, volume=volume, last_price=last_price,
                          datetime=date_time, gateway_name=gateway_name, ask_price_1=ask_price_1,
                          ask_volume_1=ask_volume_1, high_price=high_price, low_price=low_price)
            return
        else:
            self.put_tick(symbol=symbol, exchange=exchange,
                          name=name, volume=volume, last_price=last_price, high_price=high_price, low_price=low_price,
                          datetime=date_time, gateway_name=gateway_name)
            return

    def on_query_contracts(self) -> None:
        if not self.query_contract:
            self.query_contract = True
            # put all contract event to main engine
            for stock in self.all_stock_data.itertuples():
                symbol: str = stock.code
                exchange: Exchange = Exchange.SH if symbol.startswith("6") else Exchange.SZ
                name: str = stock.name
                pricetick: float = 0.01
                size: int = 100
                product: Product = Product.EQUITY
                history_data = True
                gateway_name = self.gateway_name
                contract = ContractData(symbol=symbol, exchange=exchange,
                                        name=name, pricetick=pricetick, size=size,
                                        product=product, history_data=history_data,
                                        gateway_name=gateway_name)
                self.on_contract(contract)
            for fund in self.all_fund_data.itertuples():
                symbol: int = fund.symbol
                exchange: Exchange = Exchange.JJ  # fund exchange
                name: str = fund.sname
                pricetick: float = 0.01
                size: int = 1
                product: Product = Product.FUND
                history_data = True
                gateway_name = self.gateway_name
                contract = ContractData(symbol=symbol, exchange=exchange,
                                        name=name, pricetick=pricetick, size=size,
                                        product=product, history_data=history_data,
                                        gateway_name=gateway_name)
                self.on_contract(contract)

    def put_tick(self, **kwargs):
        """There is put tick data in queue"""
        tick = TickData(**kwargs)

        self.tick_queue.put(tick)

    def close(self):
        self.disconnect()

    def disconnect(self):
        self.start_connect = False
        self.thread.join()
        self.request_all_stocks_data.join()
        disconnect_log = LogData(self.gateway_name, msg=f"{self.gateway_name}连接已断开")
        self.on_log(disconnect_log)

    def _run(self):
        """new thread that which start put event data of tick"""
        while self.start_connect:
            try:
                tick = self.tick_queue.get(timeout=1)
                self.on_tick(tick)
            except Empty:
                # ignore error of empty
                pass

    def send_order(self, req: OrderRequest) -> str:
        pass

    def cancel_order(self, req: CancelRequest):
        pass

    def query_account(self):
        pass

    def query_position(self):
        pass
