from datetime import datetime
from enum import Enum
from typing import Optional, Sequence

from mongoengine import DateTimeField, Document, DynamicDocument, FloatField, BooleanField, StringField, IntField, \
    MapField, connect
from mongoengine.context_managers import switch_collection, switch_db
from vnpy_of_python3.examples.vnpy.trader.constant import Exchange, Interval, OrderType, Status, Offset, Direction
from vnpy_of_python3.examples.vnpy.trader.object import BarData, TickData, OrderData, TradeData, ContractData, Product
from .database import BaseDatabaseManager, Driver


def init_old(_: Driver, settings: dict):
    database = settings["database"]
    host = settings["host"]
    port = settings["port"]
    username = settings["user"]
    password = settings["password"]
    authentication_source = settings["authentication_source"]

    if not username:  # if username == '' or None, skip username
        username = None
        password = None
        authentication_source = None

    connect(
        db=database,
        host=host,
        port=port,
        username=username,
        password=password,
        authentication_source=authentication_source,
    )

    return MongoManager()


# @ ADD 添加db_alias；允许继承
class DbBarData(Document):
    """
    Candlestick bar data for database storage.

    Index is defined unique with datetime, interval, symbol
    """

    symbol: str = StringField()
    exchange: str = StringField()
    datetime: datetime = DateTimeField()
    interval: str = StringField()

    volume: float = FloatField()
    open_interest: float = FloatField()
    open_price: float = FloatField()
    high_price: float = FloatField()
    low_price: float = FloatField()
    close_price: float = FloatField()

    # vnpy1.9.2 fields
    openInterest: float = FloatField()
    high: float = FloatField()
    low: float = FloatField()
    close: float = FloatField()
    open: float = FloatField()

    meta = {
        "indexes": [
            {
                "fields": ("symbol", "exchange", "datetime"),
                # "unique": True,   # True表示不能有重复索引 # ∵'datetime'有可能会重复，∴取消unique
            }
        ],
        "allow_inheritance": True,  # new 允许继承
        'strict': False,  # 取消字段检查
        "db_alias": "bar_db"
    }

    @staticmethod
    def from_bar(bar: BarData):
        """
        Generate DbBarData object from BarData.
        """
        db_bar = DbBarData()

        db_bar.symbol = bar.symbol
        db_bar.exchange = bar.exchange.value
        db_bar.datetime = bar.datetime
        db_bar.interval = bar.interval.value
        db_bar.volume = bar.volume
        db_bar.open_interest = bar.open_interest
        db_bar.open_price = bar.open_price
        db_bar.high_price = bar.high_price
        db_bar.low_price = bar.low_price
        db_bar.close_price = bar.close_price

        return db_bar

    def to_bar(self):
        """
        Generate BarData object from DbBarData.
        """
        # vnpy2版本
        if self.interval:
            bar = BarData(
                symbol=self.symbol,
                exchange=Exchange(self.exchange),
                datetime=self.datetime,
                interval=Interval(self.interval),
                volume=self.volume,
                open_interest=self.open_interest,
                open_price=self.open_price,
                high_price=self.high_price,
                low_price=self.low_price,
                close_price=self.close_price,
                gateway_name="DB",
            )
        else:
            bar = BarData(
                symbol=self.symbol,
                exchange=Exchange(self.exchange),
                datetime=self.datetime,
                interval=Interval.MINUTE,
                volume=self.volume,
                open_interest=self.openInterest,
                open_price=self.open,
                high_price=self.high,
                low_price=self.low,
                close_price=self.close,
                gateway_name="DB",
            )
        return bar


# @ ADD 添加db_alias；允许继承
# todo DbTickData没有添加vnpy1兼容
class DbTickData(Document):
    """
    Tick data for database storage.

    Index is defined unique with (datetime, symbol)
    """

    symbol: str = StringField()
    exchange: str = StringField()
    datetime: datetime = DateTimeField()

    name: str = StringField()
    volume: float = FloatField()
    open_interest: float = FloatField()
    last_price: float = FloatField()
    last_volume: float = FloatField()
    limit_up: float = FloatField()
    limit_down: float = FloatField()

    open_price: float = FloatField()
    high_price: float = FloatField()
    low_price: float = FloatField()
    close_price: float = FloatField()
    pre_close: float = FloatField()

    bid_price_1: float = FloatField()
    bid_price_2: float = FloatField()
    bid_price_3: float = FloatField()
    bid_price_4: float = FloatField()
    bid_price_5: float = FloatField()

    ask_price_1: float = FloatField()
    ask_price_2: float = FloatField()
    ask_price_3: float = FloatField()
    ask_price_4: float = FloatField()
    ask_price_5: float = FloatField()

    bid_volume_1: float = FloatField()
    bid_volume_2: float = FloatField()
    bid_volume_3: float = FloatField()
    bid_volume_4: float = FloatField()
    bid_volume_5: float = FloatField()

    ask_volume_1: float = FloatField()
    ask_volume_2: float = FloatField()
    ask_volume_3: float = FloatField()
    ask_volume_4: float = FloatField()
    ask_volume_5: float = FloatField()

    # vnpy1.9.2 fields
    lastPrice: float = FloatField()
    lastVolume: float = FloatField()
    openInterest: float = FloatField()

    upperLimit: float = FloatField()
    lowerLimit: float = FloatField()

    openPrice: float = FloatField()
    highPrice: float = FloatField()
    lowPrice: float = FloatField()
    closePrice: float = FloatField()
    preClosePrice: float = FloatField()

    bidPrice1: float = FloatField()
    bidPrice2: float = FloatField()
    bidPrice3: float = FloatField()
    bidPrice4: float = FloatField()
    bidPrice5: float = FloatField()

    askPrice1: float = FloatField()
    askPrice2: float = FloatField()
    askPrice3: float = FloatField()
    askPrice4: float = FloatField()
    askPrice5: float = FloatField()

    bidVolume1: float = FloatField()
    bidVolume2: float = FloatField()
    bidVolume3: float = FloatField()
    bidVolume4: float = FloatField()
    bidVolume5: float = FloatField()

    askVolume1: float = FloatField()
    askVolume2: float = FloatField()
    askVolume3: float = FloatField()
    askVolume4: float = FloatField()
    askVolume5: float = FloatField()

    dataSource: str = StringField()  # 数据来源（REALTIME / 1Token / EXCHANGE）
    dataType: str = StringField()  # 数据类型（tick / depth）

    meta = {
        "indexes": [
            {
                "fields": ("symbol", "exchange", "datetime"),
                # "unique": True,
            }
        ],
        "allow_inheritance": True,  # new 允许继承
        'strict': False,  # 取消字段检查
        "db_alias": "tick_db"
    }

    @staticmethod
    def from_tick(tick: TickData):
        """
        Generate DbTickData object from TickData.
        """
        db_tick = DbTickData()

        db_tick.symbol = tick.symbol
        db_tick.exchange = tick.exchange.value
        db_tick.datetime = tick.datetime
        db_tick.name = tick.name
        db_tick.volume = tick.volume
        db_tick.open_interest = tick.open_interest
        db_tick.last_price = tick.last_price
        db_tick.last_volume = tick.last_volume
        db_tick.limit_up = tick.limit_up
        db_tick.limit_down = tick.limit_down
        db_tick.open_price = tick.open_price
        db_tick.high_price = tick.high_price
        db_tick.low_price = tick.low_price
        db_tick.pre_close = tick.pre_close

        db_tick.bid_price_1 = tick.bid_price_1
        db_tick.ask_price_1 = tick.ask_price_1
        db_tick.bid_volume_1 = tick.bid_volume_1
        db_tick.ask_volume_1 = tick.ask_volume_1

        if tick.bid_price_2:
            db_tick.bid_price_2 = tick.bid_price_2
            db_tick.bid_price_3 = tick.bid_price_3
            db_tick.bid_price_4 = tick.bid_price_4
            db_tick.bid_price_5 = tick.bid_price_5

            db_tick.ask_price_2 = tick.ask_price_2
            db_tick.ask_price_3 = tick.ask_price_3
            db_tick.ask_price_4 = tick.ask_price_4
            db_tick.ask_price_5 = tick.ask_price_5

            db_tick.bid_volume_2 = tick.bid_volume_2
            db_tick.bid_volume_3 = tick.bid_volume_3
            db_tick.bid_volume_4 = tick.bid_volume_4
            db_tick.bid_volume_5 = tick.bid_volume_5

            db_tick.ask_volume_2 = tick.ask_volume_2
            db_tick.ask_volume_3 = tick.ask_volume_3
            db_tick.ask_volume_4 = tick.ask_volume_4
            db_tick.ask_volume_5 = tick.ask_volume_5

        return db_tick

    def to_tick(self):
        """
        Generate TickData object from DbTickData.
        """
        # @ ADD vnpy1.9.2的tick
        if (not self.last_price) or (not self.ask_price_1):
            tick = TickData(
                symbol=self.symbol,
                exchange=Exchange(self.exchange),
                datetime=self.datetime,
                name=self.name,
                volume=self.volume,
                open_interest=self.openInterest,
                last_price=self.lastPrice,
                last_volume=self.lastVolume,
                limit_up=self.upperLimit,
                limit_down=self.lowerLimit,
                open_price=self.openPrice,
                high_price=self.highPrice,
                low_price=self.lowPrice,
                pre_close=self.preClosePrice,
                bid_price_1=self.bidPrice1,
                ask_price_1=self.askPrice1,
                bid_volume_1=self.bidVolume1,
                ask_volume_1=self.askVolume1,
                gateway_name="DB",
            )

            if self.bidPrice2:
                tick.bid_price_2 = self.bidPrice2
                tick.bid_price_3 = self.bidPrice3
                tick.bid_price_4 = self.bidPrice4
                tick.bid_price_5 = self.bidPrice5

                tick.ask_price_2 = self.askPrice2
                tick.ask_price_3 = self.askPrice3
                tick.ask_price_4 = self.askPrice4
                tick.ask_price_5 = self.askPrice5

                tick.bid_volume_2 = self.bidVolume2
                tick.bid_volume_3 = self.bidVolume3
                tick.bid_volume_4 = self.bidVolume4
                tick.bid_volume_5 = self.bidVolume5

                tick.ask_volume_2 = self.askVolume2
                tick.ask_volume_3 = self.askVolume3
                tick.ask_volume_4 = self.askVolume4
                tick.ask_volume_5 = self.askVolume5

            return tick

        # vnpy2
        tick = TickData(
            symbol=self.symbol,
            exchange=Exchange(self.exchange),
            datetime=self.datetime,
            name=self.name,
            volume=self.volume,
            open_interest=self.open_interest,
            last_price=self.last_price,
            last_volume=self.last_volume,
            limit_up=self.limit_up,
            limit_down=self.limit_down,
            open_price=self.open_price,
            high_price=self.high_price,
            low_price=self.low_price,
            pre_close=self.pre_close,
            bid_price_1=self.bid_price_1,
            ask_price_1=self.ask_price_1,
            bid_volume_1=self.bid_volume_1,
            ask_volume_1=self.ask_volume_1,
            gateway_name="DB",
        )

        if self.bid_price_2:
            tick.bid_price_2 = self.bid_price_2
            tick.bid_price_3 = self.bid_price_3
            tick.bid_price_4 = self.bid_price_4
            tick.bid_price_5 = self.bid_price_5

            tick.ask_price_2 = self.ask_price_2
            tick.ask_price_3 = self.ask_price_3
            tick.ask_price_4 = self.ask_price_4
            tick.ask_price_5 = self.ask_price_5

            tick.bid_volume_2 = self.bid_volume_2
            tick.bid_volume_3 = self.bid_volume_3
            tick.bid_volume_4 = self.bid_volume_4
            tick.bid_volume_5 = self.bid_volume_5

            tick.ask_volume_2 = self.ask_volume_2
            tick.ask_volume_3 = self.ask_volume_3
            tick.ask_volume_4 = self.ask_volume_4
            tick.ask_volume_5 = self.ask_volume_5

        return tick


class MongoManager_old(BaseDatabaseManager):
    def load_bar_data(
            self,
            symbol: str,
            exchange: Exchange,
            interval: Interval,
            start: datetime,
            end: datetime,
    ) -> Sequence[BarData]:
        s = DbBarData.objects(
            symbol=symbol,
            exchange=exchange.value,
            interval=interval.value,
            datetime__gte=start,
            datetime__lte=end,
        )
        data = [db_bar.to_bar() for db_bar in s]
        return data

    def load_tick_data(
            self, symbol: str, exchange: Exchange, start: datetime, end: datetime
    ) -> Sequence[TickData]:
        s = DbTickData.objects(
            symbol=symbol,
            exchange=exchange.value,
            datetime__gte=start,
            datetime__lte=end,
        )
        data = [db_tick.to_tick() for db_tick in s]
        return data

    @staticmethod
    def to_update_param(d):
        return {
            "set__" + k: v.value if isinstance(v, Enum) else v
            for k, v in d.__dict__.items()
        }

    def save_bar_data(self, datas: Sequence[BarData]):
        for d in datas:
            updates = self.to_update_param(d)
            updates.pop("set__gateway_name")
            updates.pop("set__vt_symbol")
            (
                DbBarData.objects(
                    symbol=d.symbol, interval=d.interval.value, datetime=d.datetime
                ).update_one(upsert=True, **updates)
            )

    def save_tick_data(self, datas: Sequence[TickData]):
        for d in datas:
            updates = self.to_update_param(d)
            updates.pop("set__gateway_name")
            updates.pop("set__vt_symbol")
            (
                DbTickData.objects(
                    symbol=d.symbol, exchange=d.exchange.value, datetime=d.datetime
                ).update_one(upsert=True, **updates)
            )

    def get_newest_bar_data(
            self, symbol: str, exchange: "Exchange", interval: "Interval"
    ) -> Optional["BarData"]:
        s = (
            DbBarData.objects(symbol=symbol, exchange=exchange.value)
                .order_by("-datetime")
                .first()
        )
        if s:
            return s.to_bar()
        return None

    def get_newest_tick_data(
            self, symbol: str, exchange: "Exchange"
    ) -> Optional["TickData"]:
        s = (
            DbTickData.objects(symbol=symbol, exchange=exchange.value)
                .order_by("-datetime")
                .first()
        )
        if s:
            return s.to_tick()
        return None

    def clean(self, symbol: str):
        DbTickData.objects(symbol=symbol).delete()
        DbBarData.objects(symbol=symbol).delete()


# @ ADD 数据库名称
SETTING_DB_NAME = 'VnTrader_Setting_Db'
TICK_DB_NAME = 'test_VnTrader_Tick_Db'
DAILY_DB_NAME = 'VnTrader_Daily_Db'
TRADER_TRADE_DB_NAME = 'VnTrader_Trade_Db'
TRADER_ORDER_DB_NAME = 'VnTrader_Order_Db'
MINUTE_DB_NAME = 'test_VnTrader_1Min_Db'
ACTIVE_SYMBOL_DB_NAME = "VnTrader_Status_Db"
OTHERS_DB_NAME = "VnTrader_Others_Db"
# @ ADD tick & bar
from vnpy_of_python3.examples.vnpy.trader.object import ActiveSymbolData


def init(_: Driver, settings: dict):
    """Multiple database"""
    database = settings["database"]
    host = settings["host"]
    port = settings["port"]
    username = settings["user"]
    password = settings["password"]
    authentication_source = settings["authentication_source"]

    if not username:  # if username == '' or None, skip username
        username = None
        password = None
        authentication_source = None

    connect(
        db=database,
        host=host,
        port=port,
        username=username,
        password=password,
        authentication_source=authentication_source,
    )

    # @ 连接数据库
    setting_dict = dict(host=host, port=port, authentication_source=authentication_source)
    connect(TICK_DB_NAME, alias="tick_db", **setting_dict)
    connect(MINUTE_DB_NAME, alias="bar_db", **setting_dict)
    connect(TRADER_ORDER_DB_NAME, alias="order_db", **setting_dict)
    connect(TRADER_TRADE_DB_NAME, alias='trader_db', **setting_dict)
    connect(ACTIVE_SYMBOL_DB_NAME, alias="active_db", **setting_dict)
    connect(OTHERS_DB_NAME, alias="others_db", **setting_dict)
    connect(DAILY_DB_NAME, alias="daily_db", **setting_dict)
    return MongoManager()


# @ ADD 定义DbOrderData文档
class DbOrderData(Document):
    symbol: str = StringField()
    exchange: str = StringField()
    orderid: str = StringField()

    types: str = StringField()
    direction: str = StringField()
    offset: str = StringField()
    price: float = FloatField()
    volume: float = FloatField()
    traded: float = FloatField()
    status: str = StringField()
    time: str = StringField()

    meta = {
        "indexes": [
            {
                "fields": ("symbol", "exchange", "orderid",),
                "unique": True,
            }
        ],
        "db_alias": "order_db",
    }

    @staticmethod
    def from_order(order: OrderData):
        db_order = DbOrderData()
        db_order.symbol = order.symbol
        db_order.exchange = order.exchange
        db_order.type = order.type.value
        db_order.offset = order.offset.value
        db_order.price = order.price
        db_order.volume = order.volume
        db_order.traded = order.traded
        db_order.status = order.status.value
        db_order.time = order.time

        return db_order

    def to_order(self):
        order = OrderData(
            symbol=self.symbol,
            exchange=self.exchange,
            orderid=self.orderid,
            type=OrderType(self.type),
            direction=Direction(self.direction),
            offset=Offset(self.offset),
            price=self.price,
            volume=self.volume,
            traded=self.traded,
            status=Status(self.status),
            time=self.time,
            gateway_name="DB",
        )
        return order


# @ ADD 定义DbActiveSymbolData文档
class DbActiveSymbolData(Document):
    vt_symbol: str = StringField()
    active_symbol: str = StringField()
    meta = {
        "indexes": [{"fields": ("vt_symbol",), "unique": True}],
        "db_alias": "active_db"
    }

    def to_active_symbol(self) -> ActiveSymbolData:
        active_symbol = ActiveSymbolData(vt_symbol=self.vt_symbol, active_symbol=self.active_symbol)
        return active_symbol

    @staticmethod
    def from_symbol(active_data: ActiveSymbolData) -> None:
        db_symbol = DbActiveSymbolData()
        db_symbol.vt_symbol = active_data.vt_symbol
        db_symbol.active_symbol = active_data.active_symbol
        return db_symbol


class DbPlateStockSymbolData(Document):
    code: str = StringField()
    name: str = StringField()
    c_name = StringField()
    meta = {"db_alias": "others_db", "collection": "stock_industry_plate"}


class DbStockContractData(DynamicDocument):
    symbol: str = StringField()
    exchange: str = StringField(default="ASHARE")
    name: str = StringField()
    pricetick: float = FloatField()
    Size: int = IntField()
    product: str = StringField()
    min_volume: float = FloatField(default=1.0)
    stop_supported: bool = BooleanField(default=False)
    net_position: bool = BooleanField(default=False)
    option_strike: float = FloatField(default=0.0)
    option_underlying: str = StringField()
    option_type: str = StringField()
    option_expiry: str = DateTimeField()
    history_data: bool = BooleanField()

    plate: str = StringField()  # Assigning plate to contract object
    gateway_name: str = StringField(default="ASHARE")

    ipoData: str = StringField()
    outDate: str = StringField()
    Type: int = IntField()
    status = IntField()
    meta = {"db_alias": "others_db", "collection": "ashare_contract"}

    def to_contract(self):
        contract = ContractData(gateway_name=self.gateway_name,
                                symbol=self.symbol,
                                exchange=Exchange.ASHARE,
                                name=self.name, pricetick=self.pricetick, size=self.Size,
                                history_data=self.history_data, product=Product.EQUITY)
        return contract


# @ ADD 按MongoClient[DbBarData][vt_symbol]储存
class MongoManager(MongoManager_old):
    """add collection to mongo database"""
    today_date = datetime.now().strftime("%Y%m%d")

    def save_tick_data(self, datas: Sequence[TickData]):
        for d in datas:
            updates = self.to_update_param(d)
            updates.pop("set__gateway_name")
            vt_symbol = updates.pop("set__vt_symbol")

            with switch_collection(DbTickData, vt_symbol) as Db:  # new
                Db.objects(
                           symbol=d.symbol, exchange=d.exchange.value, datetime=d.datetime
                           ).update_one(upsert=True, **updates)

    def save_bar_data(self, datas: Sequence[BarData]):
        for d in datas:
            updates = self.to_update_param(d)
            updates.pop("set__gateway_name")
            vt_symbol = updates.pop("set__vt_symbol")

            with switch_collection(DbBarData, vt_symbol) as Db:  # new
                Db.objects(
                    symbol=d.symbol, exchange=d.exchange.value, datetime=d.datetime
                ).update_one(upsert=True, **updates)

    def save_order_data(self, datas: Sequence[OrderData]):
        for d in datas:
            updates = self.to_update_param(d)
            updates.pop("set__gateway_name")
            updates.pop("set__vt_symbol")
            updates.pop("set__vt_orderid")
            type_str = updates.pop("set__type")
            updates["set__types"] = type_str
            updates['set__time'] = datetime.now().strftime("%Y-%m-%d/%H:%M:%S")

            with switch_collection(DbOrderData, self.today_date) as Db:
                Db.objects(symbol=d.symbol, exchange=d.exchange.value,
                           orderid=d.orderid).update_one(upsert=True, **updates)

    def load_tick_data(self, symbol: str, exchange: Exchange, start: datetime, end: datetime, use_yield=False) -> \
            Sequence[TickData]:
        vt_symbol = f"{symbol}.{exchange.value}"
        with switch_collection(DbBarData, vt_symbol) as Db:
            s = Db.objects(datetime__gte=start, datetime__lte=end, )

        if not use_yield:
            data = [db_tick.to_tick() for db_tick in s]
            return data
        else:
            for db_tick in s:
                yield db_tick.to_tick()

    def load_bar_data(self, symbol: str, exchange: Exchange, interval: Interval, start: datetime, end: datetime,
                      use_yield=False) -> Sequence[BarData]:
        vt_symbol = f"{symbol}.{exchange.value}"
        with switch_collection(DbBarData, vt_symbol) as ct:
            s = ct.objects(classs_check=False,  # @ REMARK class_check=False可以忽略_cls字段，令vnpy1.92的也可以读取
                           datetime__gte=start, datetime__lte=end, )

            if not use_yield:
                data = [db_bar.to_bar() for db_bar in s]
                return data
            else:
                for db_bar in s:
                    yield db_bar.to_bar()

    def load_order_data(self, symbol: str, exchange: Exchange, time: str) -> Sequence[OrderData]:
        with switch_collection(DbOrderData, time) as Db:
            s = Db.objects(symbol=symbol,
                           exchange=exchange.value, )
            data = [db_order.to_order() for db_order in s]
            return data

    def save_active_symbol_data(self, datas: Sequence[ActiveSymbolData]):
        for d in datas:
            updates = self.to_update_param(d)
            with switch_collection(DbActiveSymbolData, "Status_ActiveSymbol") as Db:
                Db.objects(vt_symbol=d.vt_symbol).update_one(upsert=True,
                                                             **updates)  # It's condition of juge that it use query data

    def load_active_symbol_data(self, vt_symbol: str = None):
        """
        :vt_symbol query condition
        :return all ActiveSymbolData object if condition of symbol equal None"""
        with switch_collection(DbActiveSymbolData, "Status_ActiveSymbol") as Db:
            s = Db.objects(vt_symbol=vt_symbol)
            data = [active_data.to_active_symbol() for active_data in s]
            return data

    def clean_date_tick_data(self, symbol: str, exchange: Exchange, date: datetime):
        vt_symbol = f"{symbol}.{exchange.value}"
        with switch_collection(DbTickData, vt_symbol) as Db:
            Db.obects(class_check=False, datetime=date).delete()
            return True

    def clean_date_bar_data(self, symbol: str, exchange: Exchange, date: datetime):

        vt_symbol = f"{symbol}.{exchange.value}"
        with switch_collection(DbBarData, vt_symbol) as Db:
            Db.obects(class_check=False, datetime=date).delete()
            return True

    def save_all_the_contract_data(self, datas: Sequence):
        for d in datas:
            updates = self.to_update_param(d)
            updates.pop("set__gateway_name")
            updates.pop("set__vt_symbol")
            updates["set__Size"] = updates.pop("set__size")
            DbStockContractData.objects(symbol=d.symbol).update_one(upsert=True, **updates)

    def load_all_the_contract_data(self):
        contracts_db = DbStockContractData.objects.all()
        data = [db_contract.to_contract() for db_contract in contracts_db]
        return data

    def load_all_the_plate_data(self):
        plates_db = DbPlateStockSymbolData.objects.all()
        plates_data_dict = {plate.code: plate.c_name for plate in plates_db}
        return plates_data_dict

    def load_all_the_bars_data(self, symbol: str, exchange: Exchange):
        vt_symbol = f"{symbol}.{exchange.value}"
        with switch_collection(DbBarData, vt_symbol) as Db:
            bar_cursor = Db.objects.all()
            return [bar.to_bar() for bar in bar_cursor]

    def save_day_bar_data(self, datas: Sequence[BarData]):
        for d in datas:
            updates = self.to_update_param(d)
            updates.pop("set__gateway_name")
            vt_symbol = updates.pop("set__vt_symbol")
            with switch_db(DbBarData, "daily_db") as Db:
                with switch_collection(Db, vt_symbol) as Db:  # new
                    Db.objects(
                        symbol=d.symbol, exchange=d.exchange.value, datetime=d.datetime
                    ).update_one(upsert=True, **updates)

    def load_day_bar_data(self, symbol, start_datetime, end_datetime):
        with switch_db(DbBarData, "daily_db") as Db:
            with switch_collection(Db, f"{symbol}.ASHARE") as Db:
                s = Db.objects(  # @ REMARK class_check=False可以忽略_cls字段，令vnpy1.92的也可以读取
                    datetime__gte=start_datetime, datetime__lte=end_datetime).clear_cls_query()
                return [d.to_bar() for d in s]

    def load_last_tick_data(self, symbol, exchange):
        vt_symbol = f"{symbol}.{exchange.value}"
        with switch_collection(DbBarData, vt_symbol) as Db:
            return Db.objects().order_by("datetime")[-1].to_tick()

    def clean_day_bar_data(self, symbol, start_datetime, end_datetime):
        with switch_db(DbBarData, "daily_db") as Db:
            with switch_collection(Db, f"{symbol}.ASHARE") as Db:
                s = Db.objects(  # @ REMARK class_check=False可以忽略_cls字段，令vnpy1.92的也可以读取
                    datetime__gte=start_datetime, datetime__lte=end_datetime).clear_cls_query()
                s.delete()
        return True
