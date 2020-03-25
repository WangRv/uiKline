from copy import deepcopy
from typing import Optional, Sequence, Union, Generator
from pymongo.cursor import Cursor
from typing import Sequence, Optional

from vnpy.trader.database.mongodb import VnMongo
from vnpy.trader.object import *
from .database import DbName, Driver, convert_dict_to_object, convert_object_to_dict
import traceback
from enum import Enum
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, OperationFailure, DuplicateKeyError, InvalidDocument
import time

from vnpy.utor.utFunction import set_aliyun_security_group, retry

def init(_: Driver, settings: dict):
    """Pass the global settings to the MongoManager."""
    return MongoManager(settings)


class MongoManager(VnMongo):
    db_data_name_to_py_object = {"tick": TickData, "bar": BarData, "order": OrderData,
                                 "active": ActiveSymbolData, "contract": ContractData}

    def __init__(self, setting):
        super().__init__(setting['host'], setting['port'])

        self.data_class_object_fields = {}

    # [TickData]
    def load_tick_data(
            self,
            symbol: str,
            exchange: "Exchange",
            start: datetime,
            end: datetime,
            use_yield=False,
            interval: Interval = None
    ) -> Union[Sequence[TickData], Generator]:
        db_name = DbName.TICK_DB_NAME.value
        collection_name = f"{symbol}.{exchange.value}"

        if interval:
            flt = {"$or": [{"dataType": interval.value}, {"interval": interval.value}],
                   "datetime": {"$gte": start, "$lte": end}}
        else:
            flt = {"datetime": {"$gte": start, "$lte": end}}
        tick_cursor = self.db_query_aggregated(db_name, collection_name, flt, 'datetime', project={'_id': 0, '_cls': 0}, return_cursor=True)

        if not use_yield:
            data = [convert_dict_to_object(d, TickData, Interval.TICK) for d in tick_cursor]
            return data
        else:
            return self._yield_tick(tick_cursor, interval)

    def _yield_tick(self, tick_cursor, interval):
        for d in tick_cursor:
            yield convert_dict_to_object(d, TickData, interval)

    def save_tick_data(self, datas: Sequence[TickData]):
        for d in datas:
            updates = convert_object_to_dict(d)

            db_name = DbName.TICK_DB_NAME.value
            collection_name = f"{d.symbol}.{d.exchange.name}"
            self.db_insert(db_name, collection_name, updates)

    # [BarData]
    def load_bar_data(
            self,
            symbol: str,
            exchange: "Exchange",
            interval: Interval,
            start: datetime,
            end: datetime,
            use_yield=False
    ) -> Union[Sequence[BarData], Generator]:
        if interval == Interval.MINUTE:
            db_name = DbName.MINUTE_DB_NAME.value
        else:
            db_name = DbName.DAILY_DB_NAME.value
        collection_name = f"{symbol}.{exchange.value}"
        flt = {"datetime": {"$gte": start, "$lte": end}}
        bar_cursor = self.db_query_aggregated(db_name, collection_name, flt, 'datetime', project={'_id': 0, '_cls': 0}, return_cursor=True)

        if not use_yield:
            data = [convert_dict_to_object(d, BarData, interval) for d in bar_cursor]
            return data
        else:
            return self._yield_bar(bar_cursor, interval)

    def _yield_bar(self, bar_cursor, interval):
        for d in bar_cursor:
            yield convert_dict_to_object(d, BarData, interval)

    def save_bar_data(self, datas: Sequence["BarData"]):
        db_name = DbName.MINUTE_DB_NAME.value

        for d in datas:
            updates = convert_object_to_dict(d)

            collection_name = f"{d.symbol}.{d.exchange.name}"
            self.db_insert(db_name, collection_name, updates)

    # [Spread Data]
    def load_spread_tick_data(
            self,
            spread: "SpreadData",
            start: datetime,
            end: datetime,
            use_yield=False
    ) -> Union[Sequence[TickData], Generator]:
        db_name = DbName.SPREAD_TICK_DB_NAME.value
        collection_name = spread.name
        flt = {"datetime": {"$gte": start, "$lte": end}}
        tick_cursor = self.db_query_aggregated(db_name, collection_name, flt, 'datetime', project={'_id': 0, '_cls': 0}, return_cursor=True)

        if not use_yield:
            data = [convert_dict_to_object(d, TickData) for d in tick_cursor]
            return data
        else:
            return self._yield_spread_tick(tick_cursor)

    def _yield_spread_tick(self, tick_cursor):
        for d in tick_cursor:
            yield convert_dict_to_object(d, TickData)

    def load_spread_bar_data(
            self,
            spread: "SpreadData",
            start: datetime,
            end: datetime,
            use_yield=False
    ) -> Union[Sequence[BarData], Generator]:
        db_name = DbName.SPREAD_MINUTE_DB_NAME.value
        collection_name = spread.name
        flt = {"datetime": {"$gte": start, "$lte": end}}
        bar_cursor = self.db_query_aggregated(db_name, collection_name, flt, 'datetime', project={'_id': 0, '_cls': 0}, return_cursor=True)

        if not use_yield:
            data = [convert_dict_to_object(d, BarData) for d in bar_cursor]
            return data
        else:
            return self._yield_spread_bar(bar_cursor)

    def _yield_spread_bar(self, bar_cursor):
        for d in bar_cursor:
            yield convert_dict_to_object(d, BarData)

    def save_spread_tick_data(
            self,
            spread: "SpreadData",
            datas: Sequence["TickData"],
            update=False,
            upsert=True
    ):
        db_name = DbName.SPREAD_TICK_DB_NAME.value

        for d in datas:
            updates = convert_object_to_dict(d)

            collection_name = spread.name
            if not update:
                self.db_insert(db_name, collection_name, updates)
            else:
                flt = {"datetime": d.datetime}
                self.db_update(db_name, collection_name, flt, updates, upsert=upsert)

    def save_spread_bar_data(
            self,
            spread: "SpreadData",
            datas: Sequence["BarData"],
            update=False,
            upsert=True
    ):
        db_name = DbName.SPREAD_MINUTE_DB_NAME.value

        for d in datas:
            updates = convert_object_to_dict(d)

            collection_name = spread.name
            if not update:
                self.db_insert(db_name, collection_name, updates)
            else:
                flt = {"datetime": d.datetime}
                self.db_update(db_name, collection_name, flt, updates, upsert=upsert)

    # -------------------------------
    def save_order_data(self, order: OrderData):
        db_name = DbName.ORDER_DB_NAME.value
        collection_name = datetime.now().strftime("%Y%m%d")

        dict_ = convert_object_to_dict(order)

        now = datetime.now()
        dict_['date'] = now.strftime("%Y-%m-%d")
        dict_['time'] = now.strftime("%H:%M:%S.%f")

        self.db_insert(db_name, collection_name, dict_)

    def save_contract_data(self, contract: ContractData):
        db_name = DbName.SETTING_DB_NAME.value
        collection_name = "contract"

        dict_ = convert_object_to_dict(contract)

        flt = {"symbol": dict_["symbol"], "exchange": dict_["exchange"]}
        self.db_update(db_name, collection_name, flt, dict_, upsert=True)

    def load_contract_data(self, symbol: str, exchange: Union["Exchange", str]):
        if isinstance(exchange, Enum):
            exchange = exchange.value

        db_name = DbName.SETTING_DB_NAME.value
        collection_name = "contract"
        flt = {"symbol": symbol, "exchange": exchange}

        docs = self.db_query(db_name, collection_name, flt)
        if not docs:
            return

        dict_ = docs[0]
        contract = convert_dict_to_object(dict_, ContractData)
        return contract

    def load_active_symbol_data(self, vt_symbol: str = None):
        db_name = DbName.ACTIVE_SYMBOL_DB_NAME.value
        collection_name = "Status_ActiveSymbol"
        flt = {"vt_symbol": vt_symbol}

        active_symbol_cursor = self.db_query(db_name, collection_name, flt)
        return [convert_dict_to_object(d, ActiveSymbolData) for d in active_symbol_cursor]

    def save_active_symbol_data(self, datas: Sequence[ActiveSymbolData]):
        db_name = DbName.ACTIVE_SYMBOL_DB_NAME.value
        collection_name = "Status_ActiveSymbol"

        for d in datas:
            updates = convert_object_to_dict(d)
            self.db_insert(db_name, collection_name, updates)

    def load_all_the_contract_data(self):
        db_name = DbName.OTHERS_DB_NAME.value
        collection_name = "ashare_contract"

        data_list = self.db_query(db_name, collection_name, {}, return_cursor=False)

        for d in data_list:
            d["size"] = d.pop("Size")
        return [convert_dict_to_object(d, ContractData) for d in data_list]

    def save_all_the_contract_data(self, datas: Sequence):
        db_name = DbName.OTHERS_DB_NAME.value
        collection_name = "ashare_contract"

        for d in datas:
            updates = convert_object_to_dict(d)
            updates["Size"] = updates.pop("size")

            flt = {"symbol": d.symbol}
            self.db_update(db_name, collection_name, flt, updates, True)

    def load_all_the_plate_data(self):
        db_name = DbName.OTHERS_DB_NAME.value
        collection_name = "stock_industry"

        plates_data_list = self.db_query(db_name, collection_name, {})
        plates_data_dict = {plate.get('code'): plate.get('c_name') for plate in plates_data_list}
        return plates_data_dict

    def save_log_data(self, d: LogData):
        db_name = DbName.LOG_DB_NAME.value
        collection_name = datetime.now().strftime("%Y%m%d")

        dict_ = convert_object_to_dict(d)
        dict_['gateway_name'] = d.gateway_name
        self.db_insert(db_name, collection_name, dict_)

    # -------------------------------
    def clean(self, symbol: str):
        """清除symbol对应的tick, bar

        由于没有代码调用实例，因此暂不实现对应方法
        """
        # DbTickData.objects(symbol=symbol).delete()
        # DbBarData.objects(symbol=symbol).delete()
        pass

    def clean_date_tick_data(self, symbol: str, exchange: "Exchange", date: "datetime"):
        db_name = DbName.TICK_DB_NAME.value
        vt_symbol = f"{symbol}.{exchange.value}"

        flt = {"datetime": date}
        self.db_delete(db_name, vt_symbol, flt)
        return True

    def clean_date_bar_data(self, symbol: str, exchange: "Exchange", date: "datetime"):
        db_name = DbName.MINUTE_DB_NAME.value
        vt_symbol = f"{symbol}.{exchange.value}"
        flt = {"datetime": date}
        self.db_delete(db_name, vt_symbol, flt)
        return True

    def load_last_bar_data(self, symbol, exchange):
        vt_symbol = f"{symbol}.{exchange.value}"
        db_name = DbName.MINUTE_DB_NAME.value
        d = self.db_query(db_name, vt_symbol, {}, project={"_id": 0, "_cls": 0}, return_cursor=True).last()
        return convert_dict_to_object(d, BarData)

    def get_newest_tick_data(
            self,
            symbol: str,
            exchange: "Exchange",
    ) -> Optional["TickData"]:
        # s = (
        #     DbBarData.objects(symbol=symbol, exchange=exchange.value)
        #         .order_by("-datetime")
        #         .first()
        # )
        # if s:
        #     return s.to_bar()
        # return None
        pass

    def get_newest_bar_data(
            self,
            symbol: str,
            exchange: "Exchange",
            interval: "Interval"
    ) -> Optional["BarData"]:
        # s = (
        #     DbTickData.objects(symbol=symbol, exchange=exchange.value)
        #         .order_by("-datetime")
        #         .first()
        # )
        # if s:
        #     return s.to_tick()
        pass
