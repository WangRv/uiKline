# encoding:UTF-8
from typing import List, Dict, Type
import pyqtgraph as pg

# from vnpy.trader.ui import QtGui, QtWidgets, QtCore
# from vnpy.trader.vtObject import VtBarData
from PyQt5 import QtGui, QtWidgets, QtCore
# from .manager import BarManager
from .base import (GREY_COLOR, WHITE_COLOR, CURSOR_COLOR, BLACK_COLOR, to_int, NORMAL_FONT)
# from .axis import DatetimeAxis
# from .item import ChartItem, CandleItem, VolumeItem

pg.setConfigOptions(antialias=True)

# 窗格
class ChartWidget(pg.PlotWidget):
    MIN_BAR_COUNT = 100     # 图中显示的数据量

    def __init__(self, parent=None):
        """show bar picture"""
        super(ChartWidget, self).__init__(parent)

        # 映射字典
        self._plots = {}            # {plotName: plot}
        self._items = {}            # {itemName: item}
        self._item_plot_map = {}    # {item: plot}

        # 变量
        self._first_plot = None     # 第一个plot是用x轴连接其他plot
        self._cursor = None         # 十字光标

        self._right_ix = 0          # 图表最右的数据的ix
        self._bar_count = self.MIN_BAR_COUNT  # 图中显示的数据量

        # 初始化ui
        self._init_ui()

    # 初始化ui
    def _init_ui(self):
        # todo 设置标题
        self.setWindowTitle("ChartWidget of vn.py")

        # 创建主图layout
        self._layout = pg.GraphicsLayout()
        self._layout.setContentsMargins(10, 10, 10, 10)     # 设置边框的上 下 左 右的内边距
        self._layout.setSpacing(0)                          # 设置所有的plot的间隔
        self._layout.setBorder(color=GREY_COLOR, width=0.8)
        self._layout.setZValue(0)
        self.setCentralItem(self._layout)

        # # todo 绑定_manager给x轴
        # self._manager = BarManager()    # todo 删去统一管理的widget.manager，直接调用plot.manager
        # self._x_axis = DatetimeAxis(self._manager, orientation='bottom')    # 传入widget的_manager作为x轴的内部数据

    # -------------------------------------------------
    # 添加图层item
    def add_item(self, item_class, item_name, plot_name):
        """
        :param item_class: 图层类
        :param item_name: 图层名
        :param plot_name: 子图
        :return:
        """
        item = item_class()    # todo 创建item，并全部都传入widget._manager。 → 创建空的item._manager
        self._items[item_name] = item       # 保存到映射字典

        plot = self._plots.get(plot_name)
        if not plot:
            print(u'item {}：不存在plot {}'.format(item_name, plot_name))
            return
        plot.addItem(item)

        self._item_plot_map[item] = plot


    # 添加子图plot
    def add_plot(self, plot_name, minimum_height=80, maximum_height=None, hide_x_axis=False):
        """添加子图plot

        :param plot_name: 子图名
        :param minimum_height: 设置最小高度
        :param maximum_height: 设置最大高度
        :param hide_x_axis: 是否隐藏x轴
        :return:
        """
        # 创建plot实例
        x_axis = DatetimeAxis(BarManager(), orientation='bottom')   # plot独立x_axis

        plot = pg.PlotItem(axisItems={'bottom': x_axis})
        plot.setMenuEnabled(False)
        plot.setClipToView(True)
        plot.hideAxis('left')
        plot.showAxis('right')
        plot.setDownsampling(mode='peak')
        plot.setRange(xRange=(0, 1), yRange=(0, 1))
        plot.hideButtons()
        plot.setMinimumHeight(minimum_height)
        if maximum_height:
            plot.setMaximumHeight(maximum_height)

        # todo 隐藏x轴
        if hide_x_axis:
            plot.hideAxis("bottom")

        # todo 判断是否首个plot
        if not self._first_plot:
            self._first_plot = plot

        # 绑定viewBox
        view = plot.getViewBox()
        view.sigXRangeChanged.connect(self._update_y_range) # 变更x轴范围时更新y轴
        view.setMouseEnabled(x=True, y=False)               # 平移/缩放只对x轴有效，对y轴无效

        # 设置右边的数值轴
        right_axis = plot.getAxis('right')
        right_axis.setWidth(60)
        right_axis.tickFont = NORMAL_FONT

        # todo 如果不是first_plot，绑定x轴到first_plot
        if self._plots:
            first_plot = list(self._plots.values())[0]  # todo self._plots是无序dict，直接获取list[0]可以获取到first_plot？
            plot.setXLink(first_plot)

        # 缓存plot实例
        self._plots[plot_name] = plot

        # [Mod] 添加到layout？
        self._layout.nextRow()      # 第0行没有任何plot
        self._layout.addItem(plot)

    # 添加光标
    def add_cursor(self):
        if not self._cursor:
            self._cursor = ChartCursor(self, self._manager, self._plots, self._item_plot_map)   # todo 传入widget._manager，改为直接使用plot._manager

    # -------------------------------------------------
    # 传入历史数据
    def update_history(self, history):
        # todo 更新manager
        self._manager.update_history(history)

        # 更新每个图层的图形
        for item in self._items.values():
            item.update_history(history)

        # 更新图层数据显示范围
        self._update_plot_limits()

        self.move_to_right()

    # 传入新bar
    def update_bar(self, bar):
        """
        Update single bar data.
        """
        self._manager.update_bar(bar)

        for item in self._items.values():
            item.update_bar(bar)

        self._update_plot_limits()

        # 图表最右的data.index在最新数据的5条数据以内时，自动更新
        if self._right_ix >= (self._manager.get_count() - 5):
            self.move_to_right()

    # 传入signal
    def update_bar_signal(self, sig_data):
        self._manager.update_signal(sig_data)

    # -------------------------------------------------
    # 获取子图
    def get_plot(self, plot_name):
        """
        Get specific plot with its name.
        """
        return self._plots.get(plot_name, None)

    # 获取所有子图
    def get_all_plots(self):
        """
        Get all plot objects.
        """
        return self._plots.values()

    # 清除所有数据
    def clear_all(self):
        """
        Clear all data.
        """
        self._manager.clear_all()

        for item in self._items.values():
            item.clear_all()

        if self._cursor:
            self._cursor.clear_all()

    # -------------------------------------------------
    # 更新图层显示范围
    def _update_plot_limits(self):
        for item, plot in self._item_plot_map.items():
            min_value, max_value = item.get_y_range()

            plot.setLimits(
                xMin=-1,
                xMax=self._manager.get_count(), # todo item.manager
                yMin=min_value,
                yMax=max_value
            )

    def _update_x_range(self):
        """
        Update the x-axis range of plots.
        """
        max_ix = self._right_ix  # 最大范围
        min_ix = self._right_ix - self._bar_count  # 最小范围

        for plot in self._plots.values():
            plot.setRange(xRange=(min_ix, max_ix), padding=0)

    def _update_y_range(self):
        """
        Update the y-axis range of plots.
        """
        view = self._first_plot.getViewBox()    # todo 只使用首个plot的viewbox  【不需要修改这里，只需要对绑定的修改】

        # 获得首个plot的x轴的可见范围
        view_range = view.viewRange()           # 返回view可见范围[[xmin, xmax]， [ymin, ymax]]
        min_ix = max(0, int(view_range[0][0]))
        max_ix = min(self._manager.get_count(), int(view_range[0][1]))  # todo 最大不超过len(_manager)的范围

        # 对每个plot和其下的item都更新y轴范围
        for item, plot in self._item_plot_map.items():
            y_range = item.get_y_range(min_ix, max_ix)
            plot.setRange(yRange=y_range)

    # -------------------------------------------------
    # 键鼠操作
    # 更新范围框
    def paintEvent(self, event):
        """
        Reimplement this method of parent to update current max_ix value.
        """
        view = self._first_plot.getViewBox()
        view_range = view.viewRange()
        self._right_ix = max(0, view_range[0][1])

        super(ChartWidget, self).paintEvent(event)

    # 鼠标点击
    def keyPressEvent(self, event):
        """
        Reimplement this method of parent to move chart horizontally and zoom in/out.
        """
        if event.key() == QtCore.Qt.Key_Left:
            self._on_key_left()
        elif event.key() == QtCore.Qt.Key_Right:
            self._on_key_right()
        elif event.key() == QtCore.Qt.Key_Up:
            self._on_key_up()
        elif event.key() == QtCore.Qt.Key_Down:
            self._on_key_down()

    # 鼠标滚轮
    def wheelEvent(self, event):
        """
        Reimplement this method of parent to zoom in/out.
        """
        delta = event.angleDelta()

        if delta.y() > 0:
            self._on_key_up()
        elif delta.y() < 0:
            self._on_key_down()

    # 拖动图表
    def _on_key_left(self):
        """
        Move chart to left.
        """
        self._right_ix -= 1
        self._right_ix = max(self._right_ix, self._bar_count)

        self._update_x_range()
        self._cursor.move_left()
        self._cursor.update_info()

    def _on_key_right(self):
        """
        Move chart to right.
        """
        self._right_ix += 1
        self._right_ix = min(self._right_ix, self._manager.get_count())

        self._update_x_range()
        self._cursor.move_right()
        self._cursor.update_info()

    # 缩放图表
    def _on_key_down(self):
        """
        Zoom out the chart.
        """
        self._bar_count *= 1.2
        self._bar_count = min(int(self._bar_count), self._manager.get_count())

        self._update_x_range()
        self._cursor.update_info()

    def _on_key_up(self):
        """
        Zoom in the chart.
        """
        self._bar_count /= 1.2
        self._bar_count = max(int(self._bar_count), self.MIN_BAR_COUNT)

        self._update_x_range()
        self._cursor.update_info()

    # 移动到图表最右的数据
    def move_to_right(self):
        """
        Move chart to the most right.
        """
        self._right_ix = self._manager.get_count()
        self._update_x_range()
        self._cursor.update_info()


# 十字光标
class ChartCursor(QtCore.QObject):
    """鼠标光标"""

    def __init__(self, widget, manager, plots, item_plot_map):
        """"""
        super(ChartCursor, self).__init__()

        self._widget = widget
        self._manager = manager # todo 传入widget._manager，改为直接使用plot._manager

        self._plots = plots
        self._item_plot_map = item_plot_map

        self._x = 0
        self._y = 0
        self._plot_name = ""

        self._init_ui()
        self._connect_signal()

    # ---------------------------
    # 初始化ui
    def _init_ui(self):
        """"""
        self._init_line()
        self._init_label()
        self._init_info()

    # 创建十字光标线
    def _init_line(self):
        """
        Create line objects.
        """
        self._v_lines = {}
        self._h_lines = {}
        self._views = {}

        pen = pg.mkPen(WHITE_COLOR)  # 白线

        for plot_name, plot in self._plots.items():
            v_line = pg.InfiniteLine(angle=90, movable=False, pen=pen)
            h_line = pg.InfiniteLine(angle=0, movable=False, pen=pen)
            view = plot.getViewBox()

            for line in [v_line, h_line]:
                line.setZValue(0)
                line.hide()
                view.addItem(line)

            self._v_lines[plot_name] = v_line
            self._h_lines[plot_name] = h_line
            self._views[plot_name] = view

    # 创建XY轴的数值标签
    def _init_label(self):
        """
        Create label objects on axis.
        """
        self._y_labels = {}
        for plot_name, plot in self._plots.items():
            label = pg.TextItem(
                plot_name, fill=CURSOR_COLOR, color=BLACK_COLOR)
            label.hide()
            label.setZValue(2)
            label.setFont(NORMAL_FONT)
            plot.addItem(label, ignoreBounds=True)
            self._y_labels[plot_name] = label  # Y标签
        self._x_label = pg.TextItem(
            "datetime", fill=CURSOR_COLOR, color=BLACK_COLOR)
        self._x_label.hide()
        self._x_label.setZValue(2)
        self._x_label.setFont(NORMAL_FONT)
        plot.addItem(self._x_label, ignoreBounds=True)

    # 创建文字指向的信息
    def _init_info(self):
        """
        """
        # self._infos: Dict[str, pg.TextItem] = {}
        self._infos = {}
        for plot_name, plot in self._plots.items():
            info = pg.TextItem(
                "info",
                color=CURSOR_COLOR,
                border=CURSOR_COLOR,
                fill=BLACK_COLOR
            )
            info.hide()
            info.setZValue(2)
            info.setFont(NORMAL_FONT)
            plot.addItem(info)  # , ignoreBounds=True)
            self._infos[plot_name] = info

    # ---------------------------
    def _connect_signal(self):
        """
        Connect mouse move signal to update function.
        """
        self._widget.scene().sigMouseMoved.connect(self._mouse_moved)

    def _mouse_moved(self, evt):
        """
        Callback function when mouse is moved.
        """
        if not self._manager.get_count():
            return

        # First get current mouse point
        pos = evt

        for plot_name, view in self._views.items():
            rect = view.sceneBoundingRect()

            if rect.contains(pos):
                mouse_point = view.mapSceneToView(pos)
                self._x = to_int(mouse_point.x())
                self._y = mouse_point.y()
                self._plot_name = plot_name
                break

        # Then update cursor component
        self._update_line()
        self._update_label()
        self.update_info()

    def _update_line(self):
        """"""
        for v_line in self._v_lines.values():
            v_line.setPos(self._x)
            v_line.show()

        for plot_name, h_line in self._h_lines.items():
            if plot_name == self._plot_name:
                h_line.setPos(self._y)
                h_line.show()
            else:
                h_line.hide()

    def _update_label(self):
        """"""
        bottom_plot = list(self._plots.values())[-1]
        axis_width = bottom_plot.getAxis("right").width()
        axis_height = bottom_plot.getAxis("bottom").height()
        axis_offset = QtCore.QPointF(axis_width, axis_height)

        bottom_view = list(self._views.values())[-1]
        bottom_right = bottom_view.mapSceneToView(
            bottom_view.sceneBoundingRect().bottomRight() - axis_offset
        )

        for plot_name, label in self._y_labels.items():
            if plot_name == self._plot_name:
                label.setText(str(self._y))
                label.show()
                label.setPos(bottom_right.x(), self._y)  # 最右边 Y轴显示
            else:
                label.hide()

        dt = self._manager.get_datetime(self._x)    # todo 根据x坐标，从self.manager获得datetime （改为获得plot.manager）
        if dt:
            self._x_label.setText(dt.strftime("%Y-%m-%d %H:%M:%S"))
            self._x_label.show()
            self._x_label.setPos(self._x, bottom_right.y())  # 鼠标X,最上边
            self._x_label.setAnchor((0, 0))

    def update_info(self):
        """"""
        buf = {}

        for item, plot in self._item_plot_map.items():
            item_info_text = item.get_info_text(self._x)
            item_name = self._widget
            if plot not in buf:
                buf[plot] = item_info_text
            else:
                if item_info_text:
                    buf[plot] += ("\n\n" + item_info_text)

        for plot_name, plot in self._plots.items():
            plot_info_text = buf[plot]
            info = self._infos[plot_name]
            info.setText(plot_info_text)
            info.show()

            view = self._views[plot_name]
            #  左上角
            top_left = view.mapSceneToView(view.sceneBoundingRect().topLeft())
            info.setPos(top_left)

    # ---------------------------
    def move_right(self):
        """
        Move cursor index to right by 1.
        """
        if self._x == self._manager.get_count() - 1:    # todo 如果当前的x坐标就是manager的最后一个数据点，（无法右移），直接return
            return
        self._x += 1

        self._update_after_move()

    def move_left(self):
        """
        Move cursor index to left by 1.
        """
        if self._x == 0:
            return
        self._x -= 1

        self._update_after_move()

    def _update_after_move(self):
        """
        Update cursor after moved by left/right.
        """
        bar = self._manager.get_bar(self._x)
        self._y = bar.close_price

        self._update_line()
        self._update_label()

    # ---------------------------
    def clear_all(self):
        """
        Clear all data.
        """
        self._x = 0
        self._y = 0
        self._plot_name = ""

        for line in list(self._v_lines.values()) + list(self._h_lines.values()):
            line.hide()

        for label in list(self._y_labels.values()) + [self._x_label]:
            label.hide()

