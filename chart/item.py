# encoding:UTF-8
from abc import abstractmethod
from typing import List, Dict, Tuple
from pyqtgraph import functions as fn
import pyqtgraph as pg

# from vnpy.trader.ui import QtCore, QtGui, QtWidgets
from PyQt5 import QtWidgets, QtCore
from pyqtgraph.Qt import QtGui
# from vnpy.trader.vtObject import VtBarData

from .base import UP_COLOR, DOWN_COLOR, PEN_WIDTH, BAR_WIDTH, WHITE_COLOR, NORMAL_FONT
from .manager import BarManager
import numpy as np


# 图层基类
class ChartItem(pg.GraphicsObject):
    def __init__(self, manager):
        """"""
        super(ChartItem, self).__init__()

        self._manager = manager
        self._bar_picutures = {}  # 缓存每个bar的图形
        self._item_picuture = None

        self._up_pen = pg.mkPen(color=UP_COLOR, width=PEN_WIDTH)        # 阳线bar的画线颜色
        self._up_brush = pg.mkBrush(color=(0,0,0))                      # 阳线bar的填充颜色  # (0,0,0)是黑色
        self._down_pen = pg.mkPen(color=DOWN_COLOR, width=PEN_WIDTH)
        self._down_brush = pg.mkBrush(color=DOWN_COLOR)

        self._rect_area = None
        self.setFlag(self.ItemUsesExtendedStyleOption)

    # -----------------------------------------
    # 绘制单个bar图
    @abstractmethod
    def _draw_bar_picture(self, ix, bar):
        """
        Draw picture for specific bar.
        """
        pass

    # -----------------------------------------
    # 绘制历史bar图
    def update_history(self, history=None):
        """
        Update a list of bar data.
        Not use list of history
        """
        # 清空bar图形缓存
        self._bar_picutures.clear()  # remark 对于二次重画图时用（之前show()完一次之后关闭，过段时间后重新show()时，要清空之前的bar）

        # 从_manager获取所有bar，并绘制单期bar图形
        bars = self._manager.get_all_bars()
        for ix, bar in enumerate(bars):
            bar_picture = self._draw_bar_picture(ix, bar)
            self._bar_picutures[ix] = bar_picture

        # 更新到图上
        self.update()

    # 更新单个bar
    def update_bar(self, bar):
        """
        Update single bar data.
        """
        ix = self._manager.get_index(bar.datetime)

        bar_picture = self._draw_bar_picture(ix, bar)
        self._bar_picutures[ix] = bar_picture

        self.update()

    # 更新显示
    def update(self):
        """
        Refresh the item.
        """
        if self.scene():
            self.scene().update()

    # -----------------------------------------
    # 获取图层边框
    @abstractmethod
    def boundingRect(self):
        """
        Get bounding rectangles for item.
        """
        pass

    # 获取[min_ix, max_ix]内的y值范围
    @abstractmethod
    def get_y_range(self, min_ix=None, max_ix=None):
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
        """
        rect = opt.exposedRect

        min_ix = int(rect.left())
        max_ix = int(rect.right())
        max_ix = min(max_ix, len(self._bar_picutures))

        rect_area = (min_ix, max_ix)
        if rect_area != self._rect_area or not self._item_picuture:
            self._rect_area = rect_area
            self._draw_item_picture(min_ix, max_ix)

        self._item_picuture.play(painter)

    def _draw_item_picture(self, min_ix, max_ix):
        """
        Draw the picture of item in specific range.
        self._bar_picutures
        """
        self._item_picuture = QtGui.QPicture()
        painter = QtGui.QPainter(self._item_picuture)

        for n in range(min_ix, max_ix):
            bar_picture = self._bar_picutures[n]
            bar_picture.play(painter)

        painter.end()

    def clear_all(self):
        """
        Clear all data in the item.
        """
        self._item_picuture = None
        self._bar_picutures.clear()
        self.update()

# bar图层
class CandleItem(ChartItem):
    """单个K线图"""

    def __init__(self, manager):
        """"""
        super(CandleItem, self).__init__(manager)

    def _draw_bar_picture(self, ix, bar):
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
        """"""
        min_price, max_price = self._manager.get_price_range()
        rect = QtCore.QRectF(
            0,
            min_price,
            len(self._bar_picutures),
            max_price - min_price
        )
        return rect

    def get_y_range(self, min_ix=None, max_ix=None):
        """
        Get range of y-axis with given x-axis range.

        If min_ix and max_ix not specified, then return range with whole data set.
        """
        min_price, max_price = self._manager.get_price_range(min_ix, max_ix)
        return min_price, max_price

    def get_info_text(self, ix):
        """
        Get information text to show by cursor.
        """
        bar = self._manager.get_bar(ix)

        if bar:
            words = [
                "Date",
                bar.datetime.strftime("%Y-%m-%d"),

                "Time",
                bar.datetime.strftime("%H:%M"),

                "Open",
                # str(bar.open_price),
                str(bar.open),

                "High",
                # str(bar.high_price),
                str(bar.high),

                "Low",
                # str(bar.low_price),
                str(bar.low),

                "Close",
                # str(bar.close_price)
                str(bar.close)
            ]
            text = "\n".join(words)
        else:
            text = ""

        return text

# 柱形图图层
class VolumeItem(ChartItem):
    """成交量"""

    def __init__(self, manager):
        """"""
        super(VolumeItem, self).__init__(manager)

    def _draw_bar_picture(self, ix, bar):
        """"""
        # Create objects
        volume_picture = QtGui.QPicture()
        painter = QtGui.QPainter(volume_picture)

        # Set painter color
        if bar.close >= bar.open:
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
            bar.volume
        )
        painter.drawRect(rect)

        # Finish
        painter.end()
        return volume_picture

    def boundingRect(self):
        """"""
        min_volume, max_volume = self._manager.get_volume_range()
        rect = QtCore.QRectF(
            0,
            min_volume,
            len(self._bar_picutures),
            max_volume - min_volume
        )
        return rect

    def get_y_range(self, min_ix=None, max_ix=None):
        """
        Get range of y-axis with given x-axis range.

        If min_ix and max_ix not specified, then return range with whole data set.
        """
        min_volume, max_volume = self._manager.get_volume_range(min_ix, max_ix)
        return min_volume, max_volume

    def get_info_text(self, ix):
        """
        Get information text to show by cursor.
        """
        bar = self._manager.get_bar(ix)

        if bar:
            # text = f"Volume {bar.volume}"
            text = "Volume {}".format(bar.volume)
        else:
            text = ""

        return text

# 线形图图层
class LineItem(ChartItem):
    def __init__(self, manager):
        super(LineItem, self).__init__(manager)
        self.line_pen = pg.mkPen(color=WHITE_COLOR, width=PEN_WIDTH)
        self._update = False

    def _draw_bar_picture(self, ix, bar):
        k_line_picture = QtGui.QPicture()
        painter = QtGui.QPainter(k_line_picture)

        old_bar = self._manager.get_bar(ix - 1)

        if not old_bar:
            painter.setPen(self.line_pen)
            painter.drawLine(QtCore.QPointF(ix, bar.close),
                             QtCore.QPointF(ix, bar.close))
        else:
            painter.setPen(self.line_pen)
            painter.drawLine(QtCore.QPointF(ix - 1, old_bar.close),
                             QtCore.QPointF(ix, bar.close))
        painter.end()
        return k_line_picture

    def _draw_bar_picture_old(self, ix, bar):
        # draw k line
        if not self._manager.get_bar(ix + 1):
            next_bar = bar
            bar = self._manager.get_bar(ix - 1)
            self._update = True
        else:
            next_bar = self._manager.get_bar(ix + 1)

        k_line_picture = QtGui.QPicture()
        painter = QtGui.QPainter(k_line_picture)

        old_close = bar.close
        next_close = next_bar.close
        painter.setPen(self.line_pen)
        painter.drawLine(QtCore.QPointF(ix, old_close),
                         QtCore.QPointF(ix + 1, next_close))
        painter.end()
        return k_line_picture

    def boundingRect(self):
        """"""
        min_price, max_price = self._manager.get_price_range()
        rect = QtCore.QRectF(
            0,
            min_price,
            len(self._bar_picutures),
            max_price - min_price
        )
        return rect

    def get_y_range(self, min_ix=None, max_ix=None):
        """
        Get range of y-axis with given x-axis range.

        If min_ix and max_ix not specified, then return range with whole data set.
        """
        min_price, max_price = self._manager.get_price_range(min_ix, max_ix)
        return min_price, max_price

    def get_info_text(self, ix):
        return ""

# 箭头图图层
class ArrowItem(ChartItem):
    def __init__(self, manager):
        super(ArrowItem, self).__init__(manager)

        self._red_pen = pg.mkPen(color=UP_COLOR, width=PEN_WIDTH, brush=UP_COLOR)
        self._white_pen = pg.mkPen(color=WHITE_COLOR, width=PEN_WIDTH)
        self._green_pen = pg.mkPen(color=DOWN_COLOR, width=PEN_WIDTH, brush=DOWN_COLOR)

    def _draw_bar_picture(self, ix, bar):
        sig = getattr(bar, "direction", False)

        # 有信号，画图
        if sig:
            sig_picture = self._make_arrow_picture(ix, bar, sig)
            return sig_picture
        # 无信号，返回空图（不能直接返回None）
        else:
            arrow_picture = QtGui.QPicture()
            painter = QtGui.QPainter(arrow_picture)
            painter.end()
            return arrow_picture

    # 创建箭头图层
    def _make_arrow_picture(self, ix, bar, sig):
        arrow_picture = QtGui.QPicture()
        painter = QtGui.QPainter(arrow_picture)

        if sig == 'BK':
            painter.setPen(self._white_pen)
            brush = QtGui.QBrush(QtGui.QColor("red"))
            painter.setFont(NORMAL_FONT)
            painter.setBrush(brush)

            # 绘制箭头和文本
            text = ['BK', bar.trade_price]
            painter_path = self.makeArrowPath(ix, bar.low, "up", text)
            painter.drawPath(painter_path)

        elif sig == 'SP':
            painter.setPen(self._white_pen)
            brush = QtGui.QBrush(QtGui.QColor("gray"))
            painter.setBrush(brush)

            text = ['SP', bar.trade_price]
            painter_path = self.makeArrowPath(ix, bar.high, "down", text)
            painter.drawPath(painter_path)

        elif sig == 'SK':
            painter.setPen(self._white_pen)
            brush = QtGui.QBrush(QtGui.QColor("white"))
            painter.setBrush(brush)

            text = ['SK', bar.trade_price]
            painter_path = self.makeArrowPath(ix, bar.high, "down", text)
            painter.drawPath(painter_path)

        elif sig == 'BP':
            painter.setPen(self._white_pen)
            brush = QtGui.QBrush(QtGui.QColor("darkRed"))
            painter.setBrush(brush)
            painter_path = self.makeArrowPath(ix, bar.low, "up")
            painter.drawPath(painter_path)

            text = ['BP', bar.trade_price]
            painter_path = self.makeArrowPath(ix, bar.low, "up", text)
            painter.drawPath(painter_path)

        painter.end()
        return arrow_picture

    # 制作箭头
    def makeArrowPath(self, x=0, y=0, orientation="up", text=(), width=0.001, height=0.002):
        """制作箭头

        :param x: 顶点x轴坐标
        :param y: 顶点y轴坐标
        :param orientation: 方向
        :param width: 箭头宽度
        :param height: 箭头长度
        :return:
        """

        path = QtGui.QPainterPath()

        if orientation == "up":
            # 绘制箭头
            top = (x, y*(1-0.5*height))                 # 上箭头顶点
            left1 = (x*(1-2*width), y*(1-1*height))     # 上箭头左尖点
            left2 = (x*(1-1*width), y*(1-1*height))     # 上箭头左中间点
            left3 = (x*(1-1*width), y*(1-2*height))     # 上箭头左底点

            right1 = (x*(1+2*width), y*(1-1*height))    # 上箭头右尖点
            right2 = (x*(1+1*width), y*(1-1*height))    # 上箭头右中间点
            right3 = (x*(1+1*width), y*(1-2*height))    # 上箭头右底点

            path.moveTo(*top)
            path.lineTo(*left1)
            path.lineTo(*left2)
            path.lineTo(*left3)
            path.lineTo(*right3)
            path.lineTo(*right2)
            path.lineTo(*right1)
            path.lineTo(*top)

            # 设置文本信息
            textpos = (x*(1-3.5*width), y*(1-2*height))   # 文本位置

            textstr = '\n'.join(map(str, text))

            textItem = pg.TextItem(textstr)
            textItem.setPos(*textpos)
            view = self.getViewBox()                    # 必须使用view box添加文本 不然文本会扭曲或者颠倒
            view.addItem(textItem)

        else:
            top = (x, y*(1+0.5*height))
            left1 = (x*(1-2*width), y*(1+1*height))
            left2 = (x*(1-1*width), y*(1+1*height))
            left3 = (x*(1-1*width), y*(1+2*height))

            right1 = (x*(1+2*width), y*(1+1*height))
            right2 = (x*(1+1*width), y*(1+1*height))
            right3 = (x*(1+1*width), y*(1+2*height))

            path.moveTo(*top)
            path.lineTo(*left1)
            path.lineTo(*left2)
            path.lineTo(*left3)
            path.lineTo(*right3)
            path.lineTo(*right2)
            path.lineTo(*right1)
            path.lineTo(*top)

            # 设置文本信息
            n = 5 + len(text)   # 根据text行数来决定文本y轴位置
            textpos = (x*(1-3.5*width), y*(1+n*height))

            textstr = '\n'.join(map(str, text))

            textItem = pg.TextItem(textstr)
            textItem.setPos(*textpos)
            view = self.getViewBox()
            view.addItem(textItem)

        return path

    # ----------------------------------
    def boundingRect(self):
        """"""
        min_price, max_price = self._manager.get_price_range()
        rect = QtCore.QRectF(
            0,
            min_price,
            len(self._bar_picutures),
            max_price - min_price
        )
        return rect

    def get_y_range(self, min_ix=None, max_ix=None):
        """
        Get range of y-axis with given x-axis range.

        If min_ix and max_ix not specified, then return range with whole data set.
        """
        min_price, max_price = self._manager.get_price_range(min_ix, max_ix)
        return min_price, max_price

    def get_info_text(self, ix):
        """
        Get information text to show by cursor.
        """
        bar = self._manager.get_bar(ix)
        #
        if hasattr(bar, "trade_position"):
            #     # text = f"Volume {bar.volume}"
            text = "Order {}\n" \
                   "trade_position {}\n" \
                   "trade_price {}\n" \
                   "hold_position{}\n" \
                   "pnl{}".format(bar.order, bar.trade_position,
                                  bar.trade_price, bar.hold_position, bar.pnl)
            text = ""
            return text
        else:
            text = ""
            return text
# 均线图册
class AverageItem(pg.GraphicsItem):
    def __init__(self,manager):
        super(AverageItem, self).__init__()
        self._manager = manager
        self.line_pen = pg.mkPen(color=WHITE_COLOR,width=PEN_WIDTH)
        self.setFlag(self.ItemUsesExtendedStyleOption)
