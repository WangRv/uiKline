# encoding:UTF-8
from PyQt5 import QtCore
from collections import OrderedDict
import pyqtgraph as pg
from .base import *
import numpy as np


class AverageItem(pg.GraphicsObject):
    def __init__(self, manager):
        super(AverageItem, self).__init__()
        self._manager = manager
        self._bar_pictures = {}
        self._bar_result = {}
        self._rect_area = None
        self._bar_attribute = "close"
        self._average_interval = 5  # 5
        self._average_data_array = np.zeros(self._average_interval)
        self.line_pen = pg.mkPen(color=WHITE_COLOR, width=PEN_WIDTH)
        self.setFlag(self.ItemUsesExtendedStyleOption)

    def set_bar_attribute(self, attribute, interval=5):
        if isinstance(attribute, str) and isinstance(interval, int):
            self._bar_attribute = attribute
            self._average_interval = interval
            self._average_data_array = np.zeros(self._average_interval)

    def  old_draw_bar_picture(self, ix, bar):
        average_picture = QtGui.QPicture()
        painter = QtGui.QPainter(average_picture)
        painter.setPen(self.line_pen)
        if ix < self._average_interval:
            self._average_data_array[ix] = getattr(bar, self._bar_attribute)
            result = self._average_data_array.mean() # [1,2,3,4,5] .mean(
            self._bar_result[ix] = result
            old_result = self._bar_result.get(ix - 1)
            if not old_result:
                # first bar

                painter.drawLine(QtCore.QPointF(ix, result), QtCore.QPointF(ix, result))

            else:
                # from 2 to self.interval
                painter.drawLine(QtCore.QPointF(ix - 1, old_result), QtCore.QPointF(ix, result))
            painter.end()
            return average_picture
        else:
            # When ix greater then specify interval
            self._average_data_array[:-1] = self._average_data_array[1:]
            self._average_data_array[-1] = getattr(bar, self._bar_attribute)
            result = self._average_data_array.mean()
            self._bar_result[ix] = result
            last_result = self._bar_result.get(ix - 1)
            painter.drawLine(QtCore.QPointF(ix - 1, last_result), QtCore.QPointF(ix, result))
            painter.end()
            return average_picture

    def _draw_bar_picture(self, ix, bar):
        result = getattr(bar, self._bar_attribute) # bar.ma5
        average_picture = QtGui.QPicture()

        painter = QtGui.QPainter(average_picture)
        painter.setPen(self.line_pen)

        self._bar_result[ix] = result
        old_result = self._bar_result.get(ix - 1)
        if not old_result:
            # first bar

            painter.drawLine(QtCore.QPointF(ix, result), QtCore.QPointF(ix, result))
            painter.end()
            return average_picture
        else:
            # from 2 to self.interval
            painter.drawLine(QtCore.QPointF(ix - 1, old_result), QtCore.QPointF(ix, result))
            painter.end()
            return average_picture

    def update_history(self, history):
        self._bar_pictures.clear()
        bars = self._manager.get_all_bars()
        for ix, bar in enumerate(bars):
            bar_picture = self._draw_bar_picture(ix, bar)
            self._bar_pictures[ix] = bar_picture
        self.update()

    def update_bar(self, bar):
        ix = self._manager.get_index(bar.datetime)

        bar_picture = self._draw_bar_picture(ix, bar)
        self._bar_pictures[ix] = bar_picture

        self.update()

    def update(self):
        if self.scene():
            self.scene().update()

    def boundingRect(self):
        all_values = self._bar_result.values()
        min_result, max_result = min(all_values), max(all_values)
        rect = QtCore.QRectF(0, min_result, len(self._bar_pictures), max_result - min_result)
        return rect

    def get_y_range(self, min_ix=None, max_ix=None):
        if not min_ix:
            min_ix = 0
            max_ix = len(self._bar_result.keys()) - 1
        else:
            min_ix = int(min_ix)
            max_ix = int(max_ix)
            max_ix = max(max_ix, len(self._bar_result.keys()))
        range_of_index_pictures = list(self._bar_result.values())[min_ix:max_ix]
        return min(range_of_index_pictures), max(range_of_index_pictures)

    def get_info_text(self, ix):
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
        max_ix = min(max_ix, len(self._bar_pictures))

        rect_area = (min_ix, max_ix)
        if rect_area != self._rect_area or not self._item_picture:
            self._rect_area = rect_area
            self._draw_item_picture(min_ix, max_ix)

        self._item_picture.play(painter)

    def _draw_item_picture(self, min_ix, max_ix):
        """
        Draw the picture of item in specific range.
        self._bar_picutures
        """
        self._item_picture = QtGui.QPicture()
        painter = QtGui.QPainter(self._item_picture)

        for n in range(min_ix, max_ix):
            bar_picture = self._bar_pictures[n]
            bar_picture.play(painter)

        painter.end()

    def clear_all(self):
        """
        Clear all data in the item.
        """
        self._item_picture = None
        self._bar_picutures.clear()
        self._bar_result.clear()
        self.update()
