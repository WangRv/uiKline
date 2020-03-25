# encoding:UTF-8
# @auth Wang
# @Date 2019/10/1
import tushare as ts
from time import sleep
from datetime import datetime
from queue import Queue, Empty
from threading import Thread
from multiprocessing.dummy import Pool
from vnpy.trader.database import mongo_manager
from ...trader.gateway import BaseGateway
from ...trader.constant import Direction
from ...trader.object import \
    (TickData, SubscribeRequest, OrderRequest, CancelRequest,
     LogData, Exchange, Product, ContractData)
from pandas import DataFrame
from typing import List, Dict
from .down_all_stocks_data import extract_hs300_contracts, extract_sz50_contracts
from .control_opening_time import control_trade_time
from vnpy.utor import tushare as ts     # remark pycharm下，需要把vnpy.utor设置mark directory as -> sources root
from .test_client import TestClient


class ChinaStocksGateway(BaseGateway):
    # work day trade time
    open_trade_time: tuple = (9, 25,)
    close_trade_time: tuple = (15, 30,)

    column_name_map_dict = {}
    exchanges = [Exchange.ASHARE, Exchange.AETF]

    req_address = "tcp://47.56.102.183:2014"
    sub_address = "tcp://47.56.102.183:4102"

    def __init__(self, event_engine):
        super(ChinaStocksGateway, self).__init__(event_engine, "ASHARE")
        self.subscribe_stocks: list[str] = list()
        self.subscribe_funds: list[str] = list()
        self.all_stock_data: DataFrame = DataFrame()
        self.all_fund_data: DataFrame = DataFrame()
        self.bar_min: int = 1
        # related contract data
        self.query_contract: bool = False
        self.code_contract_plate_dict: Dict = {}
        self.code_to_contract_dict: Dict = {}
        # related tick data
        self.tick_queue: Queue = Queue()
        self.bar_queue:Queue = Queue()
        self.control_threading = Thread(target=self.control_opening)
        self.receive_bar_thread = Thread(target=self.receive_bar_data_from_rpc_server_thread)

        self.thread: Thread = Thread(target=self._run)
        self.request_all_stocks_data: Thread = Thread(target=self._request_stocks)
        # each minute queuest to trade of stock
        self.request_today_all_stocks_data: Thread = Thread(target=self._request_today_all_stocks_data)
        self.today_all_stocks_dict = {}  # It going to save today all stocks data

        self.cl = TestClient(self.bar_queue)
        # control connect
        self.start_connect: bool = False
        self.today: datetime = datetime.now()
        # multiprocessing control
        self._tasks_tick_request_pool: Pool = Pool()
        self._request_task_dict: Dict = {}  # code_name map to request task

    # 控制启动/关闭gateway
    def control_opening(self):
        while True:
            now  = datetime.now()
            if not self.start_connect and control_trade_time(now):
                self.start_connect= True
                self.create_tick_thread_function()
                self.opening_request_tick_thread()
                self.cl.start(req_address=self.req_address,sub_address=self.sub_address)
            elif self.start_connect and not control_trade_time(now):
                self.start_connect=False
                sleep(1)
                # self.cl.stop()
                # self.cl.join()
                # self.request_today_all_stocks_data.join()

                self.receive_bar_thread.join()
            else:
                sleep(1)

    def create_tick_thread_function(self):
        """There is tick thread that is created again"""
        # self.request_today_all_stocks_data = None
        # self.request_today_all_stocks_data: Thread = Thread(target=self._request_today_all_stocks_data)
        self.receive_bar_thread = None
        self.receive_bar_thread: Thread = Thread(target=self.receive_bar_data_from_rpc_server_thread)

    def opening_request_tick_thread(self):
        if self.request_today_all_stocks_data:
            self.request_today_all_stocks_data.start()
        if self.receive_bar_thread:
            self.receive_bar_thread.start()

    def close(self):
        self.disconnect()

    # 连接
    def connect(self, setting: dict) -> None:
        """not necessary setting"""
        self.today = datetime.now()
        # hour, minute = self.today.hour, self.today.minute
        # # raise error if now time is not  trade time
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
            self.cl.subscribe_topic("")
            self.cl.start(req_address=self.req_address,sub_address=self.sub_address)
            # self.request_today_all_stocks_data.start()
            # self.thread.start()
            self.receive_bar_thread.start()
            self.control_threading.start()
        self.write_log(f"中国股票市场连接成功")

    def disconnect(self):
        self.start_connect = False
        self.thread.join()
        self.request_all_stocks_data.join()
        disconnect_log = LogData(self.gateway_name, msg=f"{self.gateway_name}连接已断开")
        self.on_log(disconnect_log)

    # contract put
    def _request_stocks(self):
        """filter assign code of stock data that put it queue of tick

        1.查询数据库contract → 推送数据库contract
        2.查询tushare contract → 保存到数据库
        """
        while True:
            try:
                # All stock and fund data requested by the Tushare api
                self.query_all_plate_contract()
                contract_list = self.query_contract_from_the_mongo()
                if contract_list:
                    self.query_contract = True
                    for contract in contract_list:
                        self.on_contract_add_plate(contract)  # put contract event

                # self.all_stock_data = ts.get_day_all()
                self.all_stock_data = ts.get_today_all_old()
                # self.all_fund_data = ts.get_nav_open()
            except Exception as e:
                sleep(1)
                continue

            else:
                # self.on_query_contracts()
                self.on_save_contracts()
                # self.on_subscribe_hs300()  # add new function:subscribe market for all hs300 stocks
                break
        # request  tick data
        # while self.start_connect:
        #     try:
        #         # self._request_all_tick_data()
        #         self.put_today_tick()
        #     except Exception as e:
        #
        #         sleep(self.bar_min * 60)
        #         continue
        #     else:
        #         sleep(self.bar_min * 60)

    def on_query_contracts(self) -> None:
        if not self.query_contract:
            self.query_contract = True
            # put all contract event to main engine
            for stock in self.all_stock_data.itertuples():
                symbol: str = stock.code
                exchange: Exchange = Exchange.ASHARE
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
                exchange: Exchange = Exchange.AETF  # fund exchange
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

    def query_all_plate_contract(self):
        """从数据库查询contract板块"""
        plates_dict = mongo_manager.load_all_the_plate_data()
        self.code_contract_plate_dict.update(plates_dict)

    def on_contract_add_plate(self, contract: ContractData):
        """给contract添加plate"""
        code = contract.symbol
        plate = self.code_contract_plate_dict.get(code)
        if plate:
            contract.plate = plate
        self.code_to_contract_dict[code] = contract  # @todo cache contract in dict

        self.on_contract(contract)

    def on_save_contracts(self):
        # put all contract event to main engine
        contract_list = []
        for stock in self.all_stock_data.itertuples():
            symbol: str = stock.code
            exchange: Exchange = Exchange.ASHARE
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
            contract_list.append(contract)
        mongo_manager.save_all_the_contract_data(contract_list)
        if not self.query_contract:# First time query contract
            for contract in contract_list:
                self.on_contract_add_plate(contract)

    def query_contract_from_the_mongo(self):
        contracts_list = mongo_manager.load_all_the_contract_data()
        return contracts_list

    # tick请求：从tushare请求最新的tick
    def _request_today_all_stocks_data(self) -> None:
        """解析网上的tick数据，转为vnpy tick对象"""
        refresh_tick = Thread(target=self.refresh_tick)
        refresh_tick.setDaemon(True)
        refresh_tick.start()
        sleep(20)
        while self.start_connect:
            today_all_stocks_df = ts.df_stock_queue.get()
            for stock_data in today_all_stocks_df.itertuples():
                symbol = stock_data.code
                if symbol not in self.subscribe_stocks:
                    continue
                name = stock_data.name
                last_price = stock_data.trade
                open_price = stock_data.open
                high_price = stock_data.high
                low_price = stock_data.low
                volume = stock_data.volume
                last_trade_time = datetime.now()

                if symbol not in self.code_to_contract_dict:
                    # There is cache contract data locally
                    while self.all_stock_data.empty:
                        sleep(1)
                    contract = self.all_stock_data[self.all_stock_data.code == symbol]
                    self.code_to_contract_dict[symbol] = contract
                contract = self.code_to_contract_dict.get(symbol)
                try:
                    if isinstance(contract, DataFrame):
                        limit_up: float = contract.preprice.values[0] * 1.1
                        limit_down: float = contract.preprice.values[0] * 0.9
                    else:
                        limit_up: float = open_price * 1.1
                        limit_down: float = open_price * 0.9
                except Exception:
                    limit_up, limit_down = 0, 0
                finally:
                    tick = TickData(self.gateway_name, symbol, Exchange.ASHARE,
                                    last_trade_time, name, volume, open_price=open_price,
                                    last_price=last_price, high_price=high_price, low_price=low_price,
                                    limit_up=limit_up, limit_down=limit_down)
                    self.today_all_stocks_dict[symbol] = tick
                    self.tick_queue.put(tick)
            # self.put_today_tick()
            # sleep(1)

    def refresh_tick(self):
        while True:
            if ts.df_stock_queue.empty():
                ts.get_today_all()
                # self.put_today_tick()
            sleep(60)

    # tick查询：从mongo查询最新的tick
    def receive_bar_data_from_rpc_server_thread(self):
        # cache_dict = {}
        sleep(10)
        # while self.start_connect:
        #     for symbol, contract in self.code_to_contract_dict.items():
        #         bar = self.query_last_bar_from_the_mongo(symbol)
        #         if bar and contract:
        #             bar  = self.consist_bar_from_tick(bar,contract)
        #         if bar and not symbol in cache_dict:
        #
        #             self.on_tick(bar)
        #             cache_dict[symbol] = bar
        #         old_tick = cache_dict.get(symbol)
        #         if bar:
        #             if bar.datetime > old_tick.datetime:
        #
        #                 self.on_tick(bar)
        #                 cache_dict[symbol] = bar
        while self.start_connect:
            data = self.bar_queue.get()
            symbol = data.symbol
            contract = self.code_to_contract_dict.get(symbol)
            if contract and symbol in self.subscribe_stocks:
                bar = self.consist_bar_from_tick(data,contract)
                self.on_tick(bar)


    def query_last_bar_from_the_mongo(self, symbol: str) -> TickData:
        bar = mongo_manager.load_last_bar_data(symbol, Exchange.ASHARE)
        return bar if bar else None

    def consist_bar_from_tick(self, bar, contract):
        bar.name = contract.name
        bar.last_price = bar.close_price
        bar.bid_price_1 = 0
        bar.bid_volume_1 = 0
        bar.ask_price_1 = 0
        bar.ask_volume_1 = 0
        return bar

    # tick推送
    def _run(self):
        """new thread that which start put event data of tick"""
        while True:
            try:
                tick = self.tick_queue.get(timeout=1)
                self.on_tick(tick)
            except Empty:
                # ignore error of empty
                pass

    def put_today_tick(self):
        """把tick推送到tick_queue"""
        for code in self.subscribe_stocks:
            tick = self.today_all_stocks_dict.get(code)
            if tick:
                self.tick_queue.put(tick)

    def put_tick(self, **kwargs):
        """There is put tick data in queue"""
        tick = TickData(**kwargs)

        self.tick_queue.put(tick)

    # 订阅
    def subscribe(self, req: SubscribeRequest) -> None:
        """simpleness subscribe stock"""

        stock_code = req.symbol
        exchange = req.exchange

        if stock_code in self.subscribe_stocks or stock_code in self.subscribe_funds:
            self.write_log(f"{stock_code}股票已经订阅。")
            return
        if exchange == Exchange.ASHARE:
            self.subscribe_stocks.append(stock_code)
        else:
            self.subscribe_funds.append(stock_code)

    def on_subscribe_all_stocks(self):
        """subscribe all stocks data"""
        for stock in self.all_stock_data.itertuples():
            symbol: str = stock.code
            exchange: Exchange = Exchange.ASHARE
            req = SubscribeRequest(symbol, exchange)
            self.subscribe(req)

    def on_subscribe_hs300(self):
        hs300_contract: dict = extract_hs300_contracts()
        for stock_contract in hs300_contract.values():
            symbol, exchange = stock_contract.symbol, stock_contract.exchange
            req = SubscribeRequest(symbol, exchange)
            self.subscribe(req)

    def on_subscribe_sz50(self):
        sz50_contract: dict = extract_sz50_contracts()
        for stock_contract in sz50_contract.values():
            symbol, exchange = stock_contract.symbol, stock_contract.exchange
            req = SubscribeRequest(symbol, exchange)
            self.subscribe(req)

    # 其他
    def send_order(self, req: OrderRequest) -> str:
        print(req.__dict__)

    def cancel_order(self, req: CancelRequest):
        pass

    def query_account(self):
        pass

    def query_position(self):
        pass
