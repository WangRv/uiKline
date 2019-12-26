# encoding:UTF-8
from typing import List, Dict, Type
import pyqtgraph as pg

# from vnpy.trader.ui import QtGui, QtWidgets, QtCore
from vnpy.trader.vtObject import VtBarData
from PyQt5 import QtGui, QtWidgets, QtCore
# from .manager import BarManager
from .base import (GREY_COLOR, WHITE_COLOR, CURSOR_COLOR, BLACK_COLOR, to_int, NORMAL_FONT)
# from .axis import DatetimeAxis
# from .item import ChartItem, CandleItem, VolumeItem

pg.setConfigOptions(antialias=True)

# ------------------------------
import datetime as dt
from abc import abstractmethod
from typing import List, Dict, Tuple
from pyqtgraph import functions as fn
from pyqtgraph.Qt import QtGui

# ------------------------------

# 数据管理器
class DataManager(object):
    def __init__(self):
        self.x_axis = []
        self.y_axis = []
        self.y_range = {}

    # --------------------------------------------
    # 传入历史数据
    def setData(self, x_axis, y_axis):
        self.x_axis = x_axis
        self.y_axis = y_axis

    # 更新单个新bar
    def update(self, x, y):
        self.x_axis.append(x)
        self.y_axis.append(y)

    # --------------------------------------------
    # 获取[min_ix, max_ix]内的价格范围
    def get_y_range(self, min_ix=None, max_ix=None, isBar=False):
        # 如果没有传入x_axis，直接返回(0, 1)
        if not self.x_axis:
            return 0, 1

        # 取整到数据范围
        if not min_ix:
            min_ix = 0
            max_ix = len(self.x_axis) - 1
        else:
            min_ix = to_int(min_ix)
            max_ix = to_int(max_ix)
            max_ix = min(max_ix, len(self.x_axis))

        # 如果有缓存值，直接读取缓存值
        buf = self.y_range.get((min_ix, max_ix), None)
        if buf:
            return buf

        # y_axis是普通数据
        if not isBar:
            y_axis = self.y_axis[min_ix:max_ix+1]
            max_price = max(y_axis)
            min_price = min(y_axis)
        # y_axis是bar数据
        else:
            bar_list = self.y_axis[min_ix:max_ix+1]
            max_price = max(map(lambda bar: bar.high, bar_list))
            min_price = min(map(lambda bar: bar.low, bar_list))

        # 放入缓存
        self.y_range[(min_ix, max_ix)] = (min_price, max_price)

        return min_price, max_price


    # --------------------------------------------
    # 清除缓存
    def _clear_cache(self):
        self.y_range.clear()

    def clear_all(self):
        self.x_axis = []
        self.y_axis = []
        self.y_range.clear()

# 时间轴
class DatetimeAxis(pg.AxisItem):
    def __init__(self, manager, *args, **kwargs):
        super(DatetimeAxis, self).__init__(*args,**kwargs)

        self.manager = manager
        self.setPen(width=AXIS_WIDTH)

    # 转换datetime为str
    def tickStrings(self, values, scale, spacing):
        strings = []

        for ix in values:
            x = self.manager.x_axis[ix]

            if isinstance(x, dt.datetime):
                if x.hour:
                    s = x.strftime("%Y-%m-%d\n%H:%M:%S")
                else:
                    s = x.strftime("%Y-%m-%d")
            else:
                s = str(x)

            strings.append(s)
        return strings

# ------------------------------
# 图层基类
class ChartItem(pg.GraphicsObject):
    def __init__(self):
        """"""
        super(ChartItem, self).__init__()

        self.manager = DataManager()
        self.pictureDict = {}  # 缓存每个bar的图形
        self._item_picuture = None

        self._up_pen = pg.mkPen(color=UP_COLOR, width=PEN_WIDTH)        # 阳线bar的画线颜色
        self._up_brush = pg.mkBrush(color=(0,0,0))                      # 阳线bar的填充颜色  # (0,0,0)是黑色
        self._down_pen = pg.mkPen(color=DOWN_COLOR, width=PEN_WIDTH)
        self._down_brush = pg.mkBrush(color=DOWN_COLOR)

        self._rect_area = None
        self.setFlag(self.ItemUsesExtendedStyleOption)

    # -----------------------------------------
    # 绘制单个数据点图
    @abstractmethod
    def draw(self, ix, y):
        """
        Draw picture for specific bar.
        """
        pass

    # -----------------------------------------
    # 绘制全部数据点的图形
    def draw_all(self):
        # 清空图形缓存（用于二次重画图(show()完一次之后关闭，过段时间后重新show()时，清空之前图形）
        self.pictureDict.clear()

        # 绘制全部数据点的图形
        y_axis = self._manager.y_axis
        for ix, y in enumerate(y_axis):
            self.pictureDict[ix] = self.draw(ix, y)

        # 更新到图表
        self.update()

    # 绘制单个数据点的图形
    def draw_one(self, ix, y):
        self.pictureDict[ix] = self.draw(ix, y)

        self.update()

    # 更新显示
    def update(self):
        if self.scene():
            self.scene().update()

    # -----------------------------------------
    # 获取[min_ix, max_ix]内的y值范围
    def get_y_range(self, min_ix=None, max_ix=None):
        min_price, max_price = self.manager.get_y_range(min_ix, max_ix)
        return min_price, max_price

    # -----------------------------------------
    # 获取图层边框
    @abstractmethod
    def boundingRect(self):
        """
        Get bounding rectangles for item.
        """
        pass

    # 获取光标对应的标识内容
    @abstractmethod
    def get_info_text(self, ix):
        """
        Get information text to show by cursor.
        """
        pass

    # -----------------------------------------
    def paint(self, painter, opt, w):
        """
        Reimplement the paint method of parent class.
        This function is called by external QGraphicsView.

        重新实现父类的paint方法。
        这个函数由外部QGraphicsView调用。
        """
        rect = opt.exposedRect

        min_ix = int(rect.left())
        max_ix = int(rect.right())
        max_ix = min(max_ix, len(self.pictureDict))

        rect_area = (min_ix, max_ix)
        if rect_area != self._rect_area or not self._item_picuture:
            self._rect_area = rect_area
            self._draw_item_picture(min_ix, max_ix)

        self._item_picuture.play(painter)

    # 绘制[min_ix, max_ix]内的图层图形
    def _draw_item_picture(self, min_ix, max_ix):
        self._item_picuture = QtGui.QPicture()
        painter = QtGui.QPainter(self._item_picuture)

        for n in range(min_ix, max_ix):
            bar_picture = self.pictureDict[n]
            bar_picture.play(painter)

        painter.end()

    # -----------------------------------------
    def clear_all(self):
        """
        Clear all data in the item.
        """
        self._item_picuture = None
        self.pictureDict.clear()
        self.update()

# 蜡烛图
class BarItem(ChartItem):
    def draw(self, ix, y):
        bar = y

        # 创建实例
        candle_picture = QtGui.QPicture()
        painter = QtGui.QPainter(candle_picture)

        # 设置颜色
        if bar.close >= bar.open:
            painter.setPen(self._up_pen)
            painter.setBrush(self._up_brush)
        else:
            painter.setPen(self._down_pen)
            painter.setBrush(self._down_brush)

        # 绘制bar实体
        if bar.open == bar.close:
            painter.drawLine(
                QtCore.QPointF(ix - BAR_WIDTH, bar.open),
                QtCore.QPointF(ix + BAR_WIDTH, bar.open),
            )
        else:
            rect = QtCore.QRectF(
                ix - BAR_WIDTH,
                bar.open,
                BAR_WIDTH * 2,
                bar.close - bar.open
            )
            painter.drawRect(rect)

        # 绘制bar上下影线
        body_bottom = min(bar.open, bar.close)
        body_top = max(bar.open, bar.close)

        if bar.low < body_bottom:
            painter.drawLine(
                QtCore.QPointF(ix, bar.low),
                QtCore.QPointF(ix, body_bottom),
            )

        if bar.high > body_top:
            painter.drawLine(
                QtCore.QPointF(ix, bar.high),
                QtCore.QPointF(ix, body_top),
            )

        # Finish
        painter.end()
        return candle_picture

    def boundingRect(self):
        min_price, max_price = self.manager.get_y_range()
        rect = QtCore.QRectF(
            0,
            min_price,
            len(self.pictureDict),
            max_price - min_price
        )
        return rect

    def get_y_range(self, min_ix=None, max_ix=None):
        min_price, max_price = self.manager.get_y_range(min_ix, max_ix, isBar=True)
        return min_price, max_price

    def get_info_text(self, ix):
        """
        Get information text to show by cursor.
        """
        bar = self.manager.y_axis[ix]

        if bar:
            words = [
                "Date",
                bar.datetime.strftime("%Y-%m-%d"),

                "Time",
                bar.datetime.strftime("%H:%M"),

                "Open",
                str(bar.open),

                "High",
                str(bar.high),

                "Low",
                str(bar.low),

                "Close",
                str(bar.close)
            ]
            text = "\n".join(words)
        else:
            text = ""

        return text

# 柱形图图层
class HistItem(ChartItem):
    def draw(self, ix, y, isUp=True):
        # Create objects
        volume_picture = QtGui.QPicture()
        painter = QtGui.QPainter(volume_picture)

        # Set painter color
        if isUp:
            painter.setPen(self._up_pen)
            painter.setBrush(self._up_brush)
        else:
            painter.setPen(self._down_pen)
            painter.setBrush(self._down_brush)

        # Draw volume body
        rect = QtCore.QRectF(
            ix - BAR_WIDTH,
            0,
            BAR_WIDTH * 2,
            y
        )
        painter.drawRect(rect)

        # Finish
        painter.end()
        return volume_picture

    def boundingRect(self):
        """"""
        min_volume, max_volume = self.manager.get_y_range()
        rect = QtCore.QRectF(
            0,
            min_volume,
            len(self.pictureDict),
            max_volume - min_volume
        )
        return rect

    def get_info_text(self, ix):
        y = self.manager.y_axis[ix]

        if y:
            text = "Volume {}".format(y)
        else:
            text = ""

        return text

# ------------------------------




