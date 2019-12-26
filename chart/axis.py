# -*- coding: utf-8 -*-
from typing import List

import pyqtgraph as pg

from .manager import BarManager
# from .base import AXIS_WIDTH, NORMAL_FONT
from .base import AXIS_WIDTH

# 时间轴
class DatetimeAxis(pg.AxisItem):
    def __init__(self, manager, *args, **kwargs):
        super(DatetimeAxis, self).__init__(*args,**kwargs)

        self._manager = manager
        self.setPen(width=AXIS_WIDTH)
        # self.tickFont = NORMAL_FONT

    # 转换datetime为str
    def tickStrings(self, values, scale, spacing):
        """
        Convert original index to datetime string.
        """
        strings = []

        for ix in values:
            dt = self._manager.get_datetime(ix)

            if not dt:
                s = ""
            elif dt.hour:
                s = dt.strftime("%Y-%m-%d\n%H:%M:%S")
            else:
                s = dt.strftime("%Y-%m-%d")

            strings.append(s)

        return strings
