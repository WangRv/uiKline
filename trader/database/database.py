from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Optional, Sequence, TYPE_CHECKING
from vnpy.trader.setting import SETTINGS
# if TYPE_CHECKING:
from vnpy.trader.constant import Interval, Exchange  # noqa
from vnpy.trader.object import BarData, TickData  # noqa


class Driver(Enum):
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"


class BaseDatabaseManager(ABC):

    @abstractmethod
    def load_bar_data(
            self,
            symbol: str,
            exchange: "Exchange",
            interval: "Interval",
            start: datetime,
            end: datetime
    ) -> Sequence["BarData"]:
        pass

    @abstractmethod
    def load_tick_data(
            self,
            symbol: str,
            exchange: "Exchange",
            start: datetime,
            end: datetime
    ) -> Sequence["TickData"]:
        pass

    @abstractmethod
    def save_bar_data(
            self,
            datas: Sequence["BarData"],
    ):
        pass

    @abstractmethod
    def save_tick_data(
            self,
            datas: Sequence["TickData"],
    ):
        pass

    @abstractmethod
    def get_newest_bar_data(
            self,
            symbol: str,
            exchange: "Exchange",
            interval: "Interval"
    ) -> Optional["BarData"]:
        """
        If there is data in database, return the one with greatest datetime(newest one)
        otherwise, return None
        """
        pass

    @abstractmethod
    def get_newest_tick_data(
            self,
            symbol: str,
            exchange: "Exchange",
    ) -> Optional["TickData"]:
        """
        If there is data in database, return the one with greatest datetime(newest one)
        otherwise, return None
        """
        pass

    @abstractmethod
    def clean(self, symbol: str):
        """
        delete all records for a symbol
        """
        pass

    def save_day_bar_data(self, day_bar_list):
        pass


# [ADD] 数据库名称
class DbName(Enum):
    suffix = SETTINGS['database.suffix']

    TICK_DB_NAME = f'{suffix}VnTrader_Tick_Db'
    MINUTE_DB_NAME = f'{suffix}VnTrader_1Min_Db'
    DAILY_DB_NAME = 'VnTrader_Daily_Db'
    RENKO_DB_NAME = 'VnTrader_Renko_Db'
    SPREAD_TICK_DB_NAME = 'VnTrader_Spread_Tick_Db'
    SPREAD_MINUTE_DB_NAME = 'VnTrader_Spread_1Min_Db'

    SETTING_DB_NAME = f'VnTrader_Setting_Db'
    OTHERS_DB_NAME = f"VnTrader_Others_Db"

    ORDER_DB_NAME = f'VnTrader_Order_Db{suffix}'
    TRADE_DB_NAME = f'VnTrader_Trade_Db{suffix}'
    POSITION_DB_NAME = f'VnTrader_Position_Db{suffix}'

    ACTIVE_SYMBOL_DB_NAME = f"VnTrader_Status_Db{suffix}"
    STATUS_DB_NAME = f'VnTrader_Status_Db{suffix}'
    LOG_DB_NAME = f'VnTrader_Log_Db{suffix}'
    ERROR_DB_NAME = f'VnTrader_Error_Db{suffix}'

    QDSIGNAL_CRYPTO_DB_NAME = f'QD_Signal_Crypto_Db{suffix}'  # f'vn_okexf{suffix}'
    QDBACKTEST_CRYPTO_DB_NAME = f'QD_Backtest_Crypto_Db{suffix}'  # f'Vn_BackTesting{suffix}'

    QDSIGNAL_STOCK_DB_NAME = f'QD_Signal_Stock_Db{suffix}'
    QDBACKTEST_STOCK_NAME = f'QD_Backtest_Stock_Db{suffix}'


# [ADD] 转换vnpy1 - vnpy2数据
vnpy1_vnpy2_fields_map_tick = {
    'lastPrice': 'last_price',
    'lastVolume': 'last_volume',
    'openInterest': 'open_interest',
    'upperLimit': 'limit_up',
    'lowerLimit': 'limit_down',
    'openPrice': 'open_price',
    'highPrice': 'high_price',
    'lowPrice': 'low_price',
    'preClosePrice': 'pre_close',

    'bidPrice1': 'bid_price_1',
    'bidPrice2': 'bid_price_2',
    'bidPrice3': 'bid_price_3',
    'bidPrice4': 'bid_price_4',
    'bidPrice5': 'bid_price_5',
    'askPrice1': 'ask_price_1',
    'askPrice2': 'ask_price_2',
    'askPrice3': 'ask_price_3',
    'askPrice4': 'ask_price_4',
    'askPrice5': 'ask_price_5',
    'bidVolume1': 'bid_volume_1',
    'bidVolume2': 'bid_volume_2',
    'bidVolume3': 'bid_volume_3',
    'bidVolume4': 'bid_volume_4',
    'bidVolume5': 'bid_volume_5',
    'askVolume1': 'ask_volume_1',
    'askVolume2': 'ask_volume_2',
    'askVolume3': 'ask_volume_3',
    'askVolume4': 'ask_volume_4',
    'askVolume5': 'ask_volume_5'
}
vnpy1_vnpy2_fields_map_bar = {
    'open': 'open_price',
    'high': 'high_price',
    'low': 'low_price',
    'close': 'close_price',
    'openInterest': 'open_interest'
}


def convert_data_vnpy1_vnpy2(db_data: dict, interval: "Interval") -> dict:
    """转换vnpy1数据为vnpy2"""
    # map vnpy1 data fields to vnpy2 fields
    if interval == Interval.TICK or interval == Interval.DEPTH:
        db_data = {vnpy1_vnpy2_fields_map_tick.get(k, k): v for k, v in db_data.items()}
    else:
        db_data = {vnpy1_vnpy2_fields_map_bar.get(k, k): v for k, v in db_data.items()}

    # add field
    db_data.update({'interval': interval.value})
    return db_data


# [ADD] 转换data dict - vnpy object
data_class_object_fields = {}


def convert_dict_to_object(db_data: dict, class_object: "vnpy object", interval: Interval = None) -> "vnpy object":
    """转换data dict为vnpy object"""
    class_object_name = class_object.__name__
    class_object_fields = data_class_object_fields.get(class_object_name, [])  # [[所有字段], {Enum字段:Enum_type}]

    # 获取class_object的 所有字段 & Enum字段
    if not class_object_fields:
        fields_map = class_object.__dataclass_fields__
        class_object_fields.append(list(fields_map.keys()))
        class_object_fields.append({k: v.type for k, v in fields_map.items() if v.type and issubclass(v.type, Enum)})

        data_class_object_fields[class_object_name] = class_object_fields

    # 兼容vnpy1数据
    if class_object == TickData:
        if 'lastPrice' in db_data:
            db_data = convert_data_vnpy1_vnpy2(db_data, interval)

    elif class_object == BarData:
        if 'close' in db_data:
            db_data = convert_data_vnpy1_vnpy2(db_data, interval)

    # 转换Enum类型
    enum_fields = class_object_fields[1]
    for field_, type_ in enum_fields.items():
        v = db_data.get(field_)
        if v:
            db_data[field_] = type_(v)

    # creat object
    obj = class_object.from_unexpected_fields(**db_data)
    return obj


def convert_object_to_dict(d):
    """转换vnpy object为dict"""
    dict_ = {k: v.value if isinstance(v, Enum) else v for k, v in d.__dict__.items()}

    if "gateway_name" in dict_:
        dict_.pop("gateway_name")
    if "vt_symbol" in dict_:
        dict_.pop("vt_symbol")

    return dict_
