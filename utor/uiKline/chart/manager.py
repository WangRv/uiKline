# encoding:UTF-8
from typing import Dict, List, Tuple
from datetime import datetime

# from vnpy.trader.object import BarData
# from vnpy.trader.vtObject import VtBarData
from collections import OrderedDict
from .base import to_int
import pandas as pd


# 数据管理器
class BarManager:
    """K线管理，用来计算"""

    def __init__(self):
        self._bars = OrderedDict()      # {datetime: bar}

        self._datetime_index_map = {}   # {datetime: index}
        self._index_datetime_map = {}   # {index: datetime}

        self._price_ranges = {}         # price序列范围 min price and max price
        self._volume_ranges = {}        # volume序列范围 min and max volume

    # --------------------------------------------
    # 传入历史数据
    def update_history(self, history):
        # 传入bar到字典
        for bar in history:
            self._bars[bar.datetime] = bar

        self._bars = OrderedDict(sorted(self._bars.items(), key=lambda tp: tp[0]))  # 排序

        # 令bar的index-datetime互相关联
        ix_list = range(len(self._bars))  # bar的index list
        dt_list = self._bars.keys()  # bar的datetime list

        self._datetime_index_map = dict(zip(dt_list, ix_list))
        self._index_datetime_map = dict(zip(ix_list, dt_list))

        # 清除数据范围内的缓存
        self._clear_cache()

    # 更新单个新bar
    def update_bar(self, bar):
        dt = bar.datetime

        # 把bar插入到映射字典里
        if dt not in self._datetime_index_map:
            ix = len(self._bars)
            self._datetime_index_map[dt] = ix
            self._index_datetime_map[ix] = dt
        self._bars[dt] = bar

        # 清除缓存
        self._clear_cache()

    # 更新信号
    def update_signal(self, sigdata):
        # todo 统一为['BK', price, pos, text]
        sigdata.index = pd.to_datetime(sigdata.index)
        sigdata = sigdata[sigdata['direction'].notna()]
        for dt, sig_data in sigdata.iterrows():
            ix = self.get_index(dt)
            if ix:
                bar = self.get_bar(ix)
                # 把信号设置为bar属性
                if bar:
                    bar.order = sig_data.get("order")
                    bar.direction = sig_data.get("direction",False)
                    bar.trade_position = sig_data.get("trade_position")
                    bar.trade_price = sig_data.get("trade_price")
                    bar.hold_position = sig_data.get("hold_position")
                    bar.pnl = sig_data.get("pnl")

    # --------------------------------------------
    # bar数量
    def get_count(self):
        """
        Get total number of bars.
        """
        return len(self._bars)

    # 获取datetime对应的index
    def get_index(self, dt):
        """
        Get index with datetime.
        """
        return self._datetime_index_map.get(dt, None)

    # 获取index对应的datetime
    def get_datetime(self, ix):
        """
        Get datetime with index.
        """
        ix = to_int(ix)
        return self._index_datetime_map.get(ix, None)

    # 获取index对应的bar
    def get_bar(self, ix):
        """
        Get bar data with index.
        """
        ix = to_int(ix)
        dt = self._index_datetime_map.get(ix, None)
        if not dt:
            return None

        return self._bars[dt]

    # 获取所有bar
    def get_all_bars(self):
        """
        Get all bar data.
        """
        return list(self._bars.values())

    # 获取[min_ix, max_ix]内的价格范围
    def get_price_range(self, min_ix=None, max_ix=None):
        """
        Get price range to show within given index range.
        """
        if not self._bars:
            return 0, 1

        if not min_ix:
            min_ix = 0
            max_ix = len(self._bars) - 1
        else:
            min_ix = to_int(min_ix)
            max_ix = to_int(max_ix)
            max_ix = min(max_ix, self.get_count())

        buf = self._price_ranges.get((min_ix, max_ix), None)
        if buf:
            return buf

        bar_list = list(self._bars.values())[min_ix:max_ix + 1]  # 这里不需要+1 虽然也能切片到，但是是错误的语法
        first_bar = bar_list[0]  # 取第一个bar 初始化最大最小价格
        max_price = first_bar.high
        min_price = first_bar.low

        for bar in bar_list[1:]:
            max_price = max(max_price, bar.high)
            min_price = min(min_price, bar.low)
        self._price_ranges[(min_ix, max_ix)] = (min_price, max_price)
        return min_price, max_price

    # 获取[min_ix, max_ix]内的成交量范围
    def get_volume_range(self, min_ix=None, max_ix=None):
        """
        Get volume range to show within given index range.
        """
        if not self._bars:
            return 0, 1

        if not min_ix:
            min_ix = 0
            max_ix = len(self._bars) - 1
        else:
            min_ix = to_int(min_ix)
            max_ix = to_int(max_ix)
            max_ix = min(max_ix, self.get_count())

        buf = self._volume_ranges.get((min_ix, max_ix), None)
        if buf:
            return buf

        bar_list = list(self._bars.values())[min_ix:max_ix + 1]

        first_bar = bar_list[0]
        max_volume = first_bar.volume
        min_volume = 0

        for bar in bar_list[1:]:
            max_volume = max(max_volume, bar.volume)

        self._volume_ranges[(min_ix, max_ix)] = (min_volume, max_volume)
        return min_volume, max_volume

    # --------------------------------------------
    # 清除缓存
    def _clear_cache(self):
        """
        Clear cached range data.
        """
        self._price_ranges.clear()
        self._volume_ranges.clear()

    def clear_all(self):
        """
        Clear all data in manager.
        """
        self._bars.clear()
        self._datetime_index_map.clear()
        self._index_datetime_map.clear()

        self._clear_cache()

