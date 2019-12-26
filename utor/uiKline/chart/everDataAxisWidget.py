# encoding:UTF-8
from .widget import *
from .axis import *
from collections import OrderedDict
from itertools import chain


# left mouse button  press event
LEFT_MOUSE_PRESS = pg.QtCore.Qt.LeftButton


class EverDataWidget(ChartWidget):
    """axis x of ever data plot  """
    MIN_BAR_COUNT = 100

    def __init__(self, parent=None):
        super(EverDataWidget, self).__init__(parent)

        self._plots = OrderedDict()  # main order dict of plot object
        self._items = {}  # {itemName: item}
        self._item_plot_map = {}  # {item: plot}

        # 变量
        self._first_plot = None  # 第一个plot是用x轴连接其他plot
        self._cursor = None  # 十字光标
        self._right_ix = 0  # 图表最右的数据的ix
        self._bar_count = self.MIN_BAR_COUNT  # 图中显示的数据量
        # the common data and axis
        self._common_manager = BarManager()
        self._common_x_axis = DatetimeAxis(
            self._common_manager, orientation="bottom")

        # plot name map to plot object within alone data manager
        self._alone_plots = OrderedDict()  # {alonePlotName:dataManager}
        self._alone_plot_managers = {}  # {plotName:dataManager}
        self._alone_plot_bar_count = {}  # {plotName:number}
        # plot name map to plot object ix
        self._alone_plot_ix = {}  # alonePlotName:ix(integer)
        # plot name map to plot item
        self._alone_plot_name_items = {}  # alonePlotName:[item...]
        # set main focus plot, that it by the keyboard event will be change show range
        self._main_focus_plot = None  # common plot if value equal None
        self._init_ui()

    def _init_ui(self):
        """The alone plot be added to common plot of behind"""
        self.setWindowTitle("ChartWidget of vn.py")

        self._layout = pg.GraphicsLayout()
        self._layout.setContentsMargins(10, 10, 10, 10)  # 设置边框的上 下 左 右的内边距
        self._layout.setSpacing(1)  # 设置所有的plot的间隔
        self._layout.setBorder(color=GREY_COLOR, width=0.8)
        self._layout.setZValue(0)

        # common layout
        # self._layout.nextRow()
        self._common_layout = self._layout.addLayout(0, 0)
        self._common_layout.setSpacing(0)  # 设置所有的plot的间隔
        # self._layout.nextRow()  # next row add to alone layout
        # alone layout
        self._alone_layout = self._layout.addLayout(1, 0)
        self._alone_layout.setSpacing(0)
        self._alone_layout.hide()
        self.setCentralItem(self._layout)

    def get_alone_plot(self, plot_name):
        """
        Get alone specific  plot with its name.
        """
        return self._alone_plots.get(plot_name, None)

    def add_cursor(self):
        if not self._cursor:
            self._cursor = EverDataCursor(self, self._plots, self._alone_plots,
                                          self._common_manager, self._alone_plot_managers)

    # 更新历史纪录，可以指定某个子图
    def update_history(self, history, alone=False, alone_plot_name=None):
        if not alone:
            self._update_common_history(history)
        else:
            self._update_alone_history(history, alone_plot_name)

    def _update_common_history(self, history):
        # update common data manager history
        self._common_manager.update_history(history)
        for item in self._items.values():
            item.update_history(history)
        # update plot  range of view within limit
        self._update_plot_limits()
        # set range of view to rightmost
        self.move_to_right()

    def _update_alone_history(self, history, alone_plot_name):
        # update manager of plot data with its assign plot name
        alone_manager = self._alone_plot_managers.get(alone_plot_name)
        alone_manager.update_history(history)
        # update items which in alone plot
        for item in self._alone_plot_name_items.get(alone_plot_name):
            item.update_history(history)
        # update plot range of limit view  with assign name
        self._update_plot_limits(alone_plot_name)
        # set range of view to rightmost with plot name
        self.move_to_right(alone_plot_name)

    # 更新bar,可以指定某个子图
    def update_bar(self, bar, alone=False, alone_plot_name=None):
        """
        Update single bar data
        """
        if not alone:
            self._update_common_plot_dar(bar)
            self._update_plot_limits()
            # 图表最右的data.index在最新数据的5条数据以内时，自动更新
            if self._right_ix >= (self._common_manager.get_count() - 5):
                self.move_to_right()
        else:
            # update assign it's bar data with plot name map to object
            self._update_alone_plot_bar(alone_plot_name, bar)
            self._update_plot_limits(alone_plot_name)
            alone_right_ix = self._alone_plot_ix.get(alone_plot_name)
            alone_manger = self._alone_plot_managers.get(alone_plot_name)
            if alone_right_ix and alone_right_ix \
                    >= (alone_manger.get_count() - 5):
                self.move_to_right(alone_plot_name)

    def _update_common_plot_dar(self, bar):
        # update common bar manager
        self._common_manager.update_bar(bar)
        # update all common items data
        for item in self._items.values():
            item.update_bar(bar)

    def _update_alone_plot_bar(self, alone_plot_name, bar):
        alone_manger = self._alone_plot_managers.get(alone_plot_name)
        if alone_manger:
            alone_manger.update_bar(bar)
        alone_items = self._alone_plot_name_items.get(alone_plot_name)
        for item in alone_items:
            item.update_bar(bar)

    def update_bar_signal(self, sig_data, alone_plot_name=None):
        if not alone_plot_name:
            # update common manager bar data
            self._common_manager.update_signal(sig_data)
        else:
            # get alone plot manager
            alone_manager = self._alone_plot_managers.get(alone_plot_name)
            if alone_manager:
                alone_manager.update_signal(sig_data)

    # 给子图刷新显示范围，可以指定某个子图
    def _update_plot_limits(self, alone_plot_name=None):
        if not alone_plot_name:
            for item, plot in self._item_plot_map.items():
                min_value, max_value = item.get_y_range()

                plot.setLimits(
                    xMin=-1,
                    xMax=self._common_manager.get_count(),  # todo item.manager
                    yMin=min_value,
                    yMax=max_value
                )
        else:
            alone_items = self._alone_plot_name_items.get(alone_plot_name)
            alone_plot = self.get_alone_plot(alone_plot_name)
            alone_manager = self._alone_plot_managers.get(alone_plot_name)
            for item in alone_items:
                min_value, max_value = item.get_y_range()
                alone_plot.setLimits(
                    xMin=-1,
                    xMax=alone_manager.get_count(),
                    yMin=min_value,
                    yMax=max_value
                )

    def add_item(self, item_class, item_name, plot_name):
        if plot_name in self._plots:
            item = item_class(self._common_manager)
            self._items[item_name] = item
            plot = self._plots.get(plot_name)
            plot.addItem(item)
            self._item_plot_map[item] = plot

        elif plot_name in self._alone_plots:
            alone_plot = self.get_alone_plot(plot_name)
            alone_manager = self._alone_plot_managers.get(plot_name)
            items = self._alone_plot_name_items.setdefault(plot_name, [])
            item = item_class(alone_manager)
            items.append(item)
            # add item to alone_plot
            alone_plot.addItem(item)

    # 添加子图，可以自定义使用单独数据管理器
    def add_plot(self, plot_name, minimum_height=80,
                 maximum_height=None, hide_x_axis=False,
                 common=True):
        # use common x-axis plot
        if common:
            plot = self._made_plot(minimum_height, maximum_height,
                                   hide_x_axis)
            # store plot of data manager object in dict
            self._plots[plot_name] = plot
            self._common_layout.nextRow()  # 第0行没有任何plot
            self._common_layout.addItem(plot)

        else:
            # initialization alone manager that it add to this plot
            plot = self._made_plot(minimum_height, maximum_height,
                                   hide_x_axis, common, plot_name)
            # map alone plot name to alone plot object
            count_number = self.MIN_BAR_COUNT
            self._alone_plot_bar_count[plot_name] = count_number
            self._alone_plots[plot_name] = plot
            self._alone_layout.nextRow()
            self._alone_layout.addItem(plot)
            self._alone_layout.show()

    # 制作plot 函数
    def _made_plot(self, minimum_height=80,
                   maximum_height=None, hide_x_axis=False,
                   common=True, alone_plot_name=None):
        if common:
            # common plot
            plot = pg.PlotItem(axisItems={"bottom": self._common_x_axis})
            # set configuration
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

            if hide_x_axis:
                plot.hideAxis("bottom")

            if not self._first_plot:
                self._first_plot = plot

            # Connect view change signal to update y range function
            view = plot.getViewBox()
            view.sigXRangeChanged.connect(self._update_y_range)
            view.setMouseEnabled(x=True, y=False)

            # Set right axis
            right_axis = plot.getAxis('right')
            right_axis.setWidth(60)
            right_axis.tickFont = NORMAL_FONT

            # Connect x-axis link
            if self._plots:
                plot.setXLink(self._first_plot)
            return plot
        else:
            # made alone plot
            alone_manger = BarManager()
            self._alone_plot_managers[alone_plot_name] = alone_manger
            x_axis = DatetimeAxis(alone_manger, orientation="bottom")
            plot = pg.PlotItem(axisItems={"bottom": x_axis})
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

            if hide_x_axis:
                plot.hideAxis("bottom")

            # Connect view change signal to update y range function
            view = plot.getViewBox()
            view.sigXRangeChanged.connect(lambda: self._alone_update_y_range(alone_plot_name))
            view.setMouseEnabled(x=True, y=False)

            # Set right axis
            right_axis = plot.getAxis('right')
            right_axis.setWidth(60)
            right_axis.tickFont = NORMAL_FONT

            return plot

    # x and y range
    def _update_x_range(self, alone_plot_name=None):
        if not alone_plot_name:
            max_ix = self._right_ix  # 最大范围
            min_ix = self._right_ix - self._bar_count  # 最小范围

            for plot in self._plots.values():
                plot.setRange(xRange=(min_ix, max_ix), padding=0)
        else:
            alone_plot = self.get_alone_plot(alone_plot_name)
            max_ix = self._alone_plot_ix.get(alone_plot_name)
            min_ix = max_ix - self._alone_plot_bar_count.get(alone_plot_name)
            alone_plot.setRange(xRange=(min_ix, max_ix,), padding=0)

    def _update_y_range(self):

        view = self._first_plot.getViewBox()  # todo 只使用首个plot的viewbox  【不需要修改这里，只需要对绑定的修改】

        # 获得首个plot的x轴的可见范围
        view_range = view.viewRange()  # 返回view可见范围[[xmin, xmax]， [ymin, ymax]]
        min_ix = max(0, int(view_range[0][0]))
        max_ix = min(self._common_manager.get_count(), int(view_range[0][1]))  # todo 最大不超过len(_manager)的范围

        # 对每个plot和其下的item都更新y轴范围
        for item, plot in self._item_plot_map.items():
            y_range = item.get_y_range(min_ix, max_ix)
            plot.setRange(yRange=y_range)

    def _alone_update_y_range(self, alone_plot_name):
        alone_plot = self.get_alone_plot(alone_plot_name)
        alone_manger = self._alone_plot_managers.get(alone_plot_name)
        alone_items = self._alone_plot_name_items.get(alone_plot_name)
        alone_plot_view = alone_plot.getViewBox()
        view_range = alone_plot_view.viewRange()
        min_ix = max(0, int(view_range[0][0]))
        max_ix = min(alone_manger.get_count(), int(view_range[0][1]))

        for item in alone_items:
            y_range = item.get_y_range(min_ix, max_ix)
            alone_plot.setRange(yRange=y_range)

    # 键鼠操作
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Left:
            self._on_key_left(event)
        elif event.key() == QtCore.Qt.Key_Right:
            self._on_key_right(event)
        elif event.key() == QtCore.Qt.Key_Up:
            self._on_key_up(event)
        elif event.key() == QtCore.Qt.Key_Down:
            self._on_key_down(event)
       # mouse wheel

    def wheelEvent(self, event):
        delta = event.angleDelta()
        if delta.y() > 0:
            # whell up
            self._on_key_up(event)
        elif delta.y() < 0:
            self._on_key_down(event)

    # 焦点图设置
    def mouseReleaseEvent(self, ev):
        super(EverDataWidget, self).mouseReleaseEvent(ev)
        if ev.button() == LEFT_MOUSE_PRESS:
            for plot_name, plot in self._plots.items():
                if self._plot_contains_pos(plot, ev.pos()):
                    self._main_focus_plot = None
                    return
            for plot_name, plot in self._alone_plots.items():
                if self._plot_contains_pos(plot, ev.pos()):
                    self._main_focus_plot = plot_name
                    return

    def _plot_contains_pos(self, plot, pos):
        """adjuge mouse position in plot"""
        view_box = plot.getViewBox()
        rect = view_box.sceneBoundingRect()
        if rect.contains(pos):
            return True
        else:
            return False

    # 拖动图表
    def _on_key_left(self, event):
        """
        Move chart to left.
        """
        # move common x-axis plot if facous is None
        if not self._main_focus_plot:
            self._right_ix -= 1
            self._right_ix = max(self._right_ix, self._bar_count)

            self._update_x_range()
        # judge whether plot contains coordinates position
        else:
            # moved alone x-axis plot view range
            self._alone_plot_ix[self._main_focus_plot] -= 1
            self._alone_plot_ix[self._main_focus_plot] = \
                max(self._alone_plot_ix[self._main_focus_plot],
                    self._bar_count)
            self._update_x_range(self._main_focus_plot)

        self._cursor.move_left()
        self._cursor.update_info()

    def _on_key_right(self, event):
        """
        Move chart to right.
        """
        if not self._main_focus_plot:
            self._right_ix += 1
            self._right_ix = max(self._right_ix, self._bar_count)

            self._update_x_range()
            # judge whether plot contains coordinates position
        else:
            # moved alone x-axis plot view range
            self._alone_plot_ix[self._main_focus_plot] += 1
            self._alone_plot_ix[self._main_focus_plot] \
                = max(self._alone_plot_ix[self._main_focus_plot],
                      self._bar_count)

            self._update_x_range(self._main_focus_plot)
        self._cursor.move_right()
        self._cursor.update_info()

    # 缩放图表
    def _on_key_down(self, event):
        """
        Zoom out the chart.
        """
        if not self._main_focus_plot:
            self._bar_count *= 1.2
            self._bar_count = min(int(self._bar_count), self._common_manager.get_count())
            if self._bar_count == 0:
                self._bar_count = 5
            self._update_x_range()
        else:
            alone_manager = \
                self._alone_plot_managers.get(self._main_focus_plot)
            manager_count = alone_manager.get_count()
            self._alone_plot_bar_count[self._main_focus_plot] *= 1.2
            bar_count = \
                min(int(self._alone_plot_bar_count[self._main_focus_plot]),
                    manager_count)
            if bar_count == 0:
                bar_count = 5
            self._alone_plot_bar_count[self._main_focus_plot] = bar_count
            self._update_x_range(self._main_focus_plot)
            # self._cursor.update_info()

    def _on_key_up(self, event):
        """
        Zoom in the chart.
        """
        if not self._main_focus_plot:
            self._bar_count /= 1.2
            self._bar_count = min(int(self._bar_count), self._common_manager.get_count())

            self._update_x_range()
        else:
            alone_manager = \
                self._alone_plot_managers.get(self._main_focus_plot)
            manager_count = alone_manager.get_count()
            self._alone_plot_bar_count[self._main_focus_plot] /= 1.2
            bar_count = \
                min(int(self._alone_plot_bar_count[self._main_focus_plot]),
                    manager_count)
            self._alone_plot_bar_count[self._main_focus_plot] = bar_count
            self._update_x_range(self._main_focus_plot)

    # 移动到图表最右的数据
    def move_to_right(self, alone_plot_name=None):
        """
        Move chart to the most right.
        """
        if not alone_plot_name:
            self._right_ix = self._common_manager.get_count()
            self._update_x_range()
            self._cursor.update_info()
        else:
            alone_manger = self._alone_plot_managers.get(alone_plot_name)
            alone_plot_right_ix = alone_manger.get_count()
            self._alone_plot_ix[alone_plot_name] = alone_plot_right_ix
            self._update_x_range(alone_plot_name)

    def get_focus_plot_name(self):
        return self._main_focus_plot

    def get_common_plot_map_items(self):
        return self._item_plot_map.items()
    # drag view set ever alone plot show range
    def paintEvent(self, event):
        for alone_name, plot_object in self._alone_plots.items():
            view = plot_object.getViewBox()
            view_range = view.viewRange()
            self._alone_plot_ix[alone_name] = max(0, view_range[0][1])

        super(EverDataWidget, self).paintEvent(event)

# Cursor Object
class EverDataCursor(QtCore.QObject):
    def __init__(self, widget, common_plots, alone_plots,
                 common_manager, alone_managers):
        super(EverDataCursor, self).__init__()
        self._widget = widget

        self._common_manager = common_manager  # data manager object
        self._alone_manages = alone_managers  # all alone manager dict
        self._common_plots = common_plots
        self._alone_plots = alone_plots

        self._is_common_plot = False  # value equal True if mouse position in common plot
        self._x = 0
        self._y = 0
        self._plot_name = ""

        self._init_ui()
        self._connect_signal()  # mouse event signal

    def _connect_signal(self):
        """
        Connect mouse move signal to update function.
        """
        self._widget.scene().sigMouseMoved.connect(self.mouse_moved)
        # ---------------------------
        # 初始化ui

    def _init_ui(self):
        """"""
        self._init_line()
        self._init_label()
        self._init_info()

    # line object
    def _init_line(self):
        """
            Create line objects.
            """
        self._v_common_lines = OrderedDict()
        self._v_alone_lines = OrderedDict()

        self._h_common_lines = OrderedDict()
        self._h_alone_lines = OrderedDict()

        self._common_views = OrderedDict()
        self._alone_views = OrderedDict()

        pen = pg.mkPen(WHITE_COLOR)  # 白线
        # common plot add line
        for common_plot_name, plot in self._common_plots.items():
            v_line = pg.InfiniteLine(angle=90, movable=False, pen=pen)
            h_line = pg.InfiniteLine(angle=0, movable=False, pen=pen)
            view = plot.getViewBox()

            for line in [v_line, h_line]:
                line.setZValue(0)
                line.hide()
                view.addItem(line)

            self._v_common_lines[common_plot_name] = v_line
            self._h_common_lines[common_plot_name] = h_line
            self._common_views[common_plot_name] = view
        # alone plots add line
        for alone_plot_name, plot in self._alone_plots.items():
            v_line = pg.InfiniteLine(angle=90, movable=False, pen=pen)
            h_line = pg.InfiniteLine(angle=0, movable=False, pen=pen)
            view = plot.getViewBox()

            for line in [v_line, h_line]:
                line.setZValue(0)
                line.hide()
                view.addItem(line)

            self._v_alone_lines[alone_plot_name] = v_line
            self._h_alone_lines[alone_plot_name] = h_line
            self._alone_views[alone_plot_name] = view

    # create tip and label
    def _init_label(self):
        self._y_common_plot_labels = {}
        self._y_alone_plot_labels = {}

        self._x_common_plot_labels = {}
        self._x_alone_plot_labels = {}
        # create x-label of common_plots
        common_x_label = pg.TextItem(
            "datetime", fill=CURSOR_COLOR, color=BLACK_COLOR)
        common_x_label.hide()
        common_x_label.setZValue(2)
        common_x_label.setFont(NORMAL_FONT)

        for plot_name, plot in self._common_plots.items():
            label = pg.TextItem(
                plot_name, fill=CURSOR_COLOR, color=BLACK_COLOR)
            label.hide()
            label.setZValue(2)
            label.setFont(NORMAL_FONT)
            plot.addItem(label, ignoreBounds=True)
            self._y_common_plot_labels[plot_name] = label  # Y标签
            self._x_common_plot_labels[plot_name] = common_x_label  # X标签
        plot.addItem(common_x_label, ignoreBounds=True)

        for plot_name, plot in self._alone_plots.items():
            alone_x_label = pg.TextItem(
                "datetime", fill=CURSOR_COLOR, color=BLACK_COLOR)
            alone_x_label.hide()
            alone_x_label.setZValue(2)
            alone_x_label.setFont(NORMAL_FONT)

            label = pg.TextItem(
                plot_name, fill=CURSOR_COLOR, color=BLACK_COLOR)
            label.hide()
            label.setZValue(2)
            label.setFont(NORMAL_FONT)
            plot.addItem(label, ignoreBounds=True)
            plot.addItem(alone_x_label, ignoreBounds=True)

            self._y_alone_plot_labels[plot_name] = label  # Y标签
            self._x_alone_plot_labels[plot_name] = alone_x_label  # X标签

    # create text of plots
    def _init_info(self):
        self._common_infos = {}
        self._alone_infos = {}

        for plot_name, plot in self._common_plots.items():
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
            self._common_infos[plot_name] = info

        for plot_name, plot in self._alone_plots.items():
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
            self._alone_infos[plot_name] = info

    # mouse moved event handle function
    def mouse_moved(self, ev):
        pos = ev
        for plot_name, view in self._common_views.items():
            rect = view.sceneBoundingRect()
            if rect.contains(pos):
                mouse_point = view.mapSceneToView(pos)
                self._x = to_int(mouse_point.x())
                self._y = mouse_point.y()
                self._plot_name = plot_name
                self._is_common_plot = True
                break

        for plot_name, view in self._alone_views.items():
            rect = view.sceneBoundingRect()
            if rect.contains(pos):
                mouse_point = view.mapSceneToView(pos)
                self._x = to_int(mouse_point.x())
                self._y = mouse_point.y()
                self._plot_name = plot_name
                self._is_common_plot = False  # mouse not in common plot

                break
        # Then update cursor component
        self._update_line()
        self._update_label()
        self.update_info()
        view.update()

    # update line position
    def _update_line(self):
        # moved line to view
        if self._is_common_plot:

            for alone_h_line in self._h_alone_lines.values():
                alone_h_line.hide()
            for alone_v_line in self._v_alone_lines.values():
                alone_v_line.hide()

            for v_line in self._v_common_lines.values():
                v_line.show()
                v_line.setPos(self._x)

            for plot_name, h_line in self._h_common_lines.items():
                if plot_name == self._plot_name:
                    h_line.setPos(self._y)
                    h_line.show()
                else:
                    h_line.hide()
        # moved line to alone plot
        else:
            # hide all common line
            for _, common_line in chain(self._v_common_lines.items(),
                                        self._h_common_lines.items()):
                common_line.hide()
            for _, alone_line in chain(self._v_alone_lines.items(),
                                       self._h_alone_lines.items()):
                alone_line.hide()

            alone_show_v = self._v_alone_lines.get(self._plot_name)
            alone_show_h = self._h_alone_lines.get(self._plot_name)
            if alone_show_v and alone_show_h:
                alone_show_h.setPos(self._y)
                alone_show_v.setPos(self._x)

                alone_show_h.show()
                alone_show_v.show()

    def _update_label(self):
        """"""
        if self._is_common_plot:
            # hide all alone label of plot
            [label.hide() for label in self._x_alone_plot_labels.values()]
            [label.hide() for label in self._y_alone_plot_labels.values()]

            bottom_plot = list(self._common_plots.values())[-1]
            axis_width = bottom_plot.getAxis("right").width()
            axis_height = bottom_plot.getAxis("bottom").height()
            axis_offset = QtCore.QPointF(axis_width, axis_height)

            bottom_view = list(self._common_views.values())[-1]
            bottom_right = bottom_view.mapSceneToView(
                bottom_view.sceneBoundingRect().bottomRight() - axis_offset
            )

            for plot_name, label in self._y_common_plot_labels.items():
                if self._plot_name == plot_name:
                    label.setText(str(self._y))
                    label.show()
                    label.setPos(bottom_right.x(), self._y)
                else:
                    label.hide()
            dt = self._common_manager.get_datetime(self._x)
            if not dt:
                return
            if dt:
                x_common_labels = list(self._x_common_plot_labels.values())[0]
                x_common_labels.setText(dt.strftime("%Y-%m-%d %H:%M:%S"))
                x_common_labels.show()
                x_common_labels.setPos(self._x, bottom_right.y())
                x_common_labels.setAnchor((0, 0))
        else:
            # hide all common label
            for common_label in chain(self._x_common_plot_labels.values(),
                                      self._y_common_plot_labels.values()):
                common_label.hide()
            # hide all alone label then show label within plot name
            [label.hide() for label in self._x_alone_plot_labels.values()]
            [label.hide() for label in self._y_alone_plot_labels.values()]
            # show label of alone plot with name
            x_alone_label = self._x_alone_plot_labels.get(self._plot_name)
            y_alone_label = self._y_alone_plot_labels.get(self._plot_name)
            if not x_alone_label or not y_alone_label:
                return
            alone_plot = self._alone_plots.get(self._plot_name)
            alone_view = self._alone_views.get(self._plot_name)
            alone_manager = self._alone_manages.get(self._plot_name)
            if not alone_plot:
                return
            axis_width = alone_plot.getAxis("right").width()
            axis_height = alone_plot.getAxis("bottom").height()
            axis_offset = QtCore.QPointF(axis_width, axis_height)

            bottom_right = alone_view.mapSceneToView(
                alone_view.sceneBoundingRect().bottomRight() - axis_offset
            )
            x_alone_label.setText(str(self._y))
            x_alone_label.show()
            x_alone_label.setPos(bottom_right.x(), self._y)

            dt = alone_manager.get_datetime(self._x)
            if not dt:
                return
            y_alone_label.setText(dt.strftime("%Y-%m-%d %H:%M:%S"))
            y_alone_label.show()
            y_alone_label.setPos(self._x, bottom_right.y())
            y_alone_label.setAnchor((0, 0))

    def update_info(self):
        """"""
        if self._is_common_plot:
            [info.hide() for info in self._alone_infos.values()]
            buf = {}
            # get all text of common plot item
            for item, plot in self._widget.get_common_plot_map_items():
                item_info_text = item.get_info_text(self._x)
                if plot not in buf:
                    buf[plot] = item_info_text
                else:
                    if item_info_text:
                        buf[plot] += ("\n\n" + item_info_text)
            # info message  it position is top-left with show
            for plot_name, plot in self._common_plots.items():
                plot_info_text = buf[plot]
                info_object = self._common_infos.get(plot_name)
                info_object.setText(plot_info_text)
                info_object.show()

                # set position
                view = self._common_views.get(plot_name)
                if not view:
                    return
                top_left = view.mapSceneToView(view.sceneBoundingRect().topLeft())
                info_object.setPos(top_left)

    # set show range after keyboard event
    def move_right(self):
        # set  plot show range
        self._x += 1
        self._update_after_move()

    def move_left(self):
        self._x -= 1
        self._update_after_move()

    def _update_after_move(self):
        # set common plot of show range
        if self._is_common_plot:
            bar = self._common_manager.get_bar(self._x)
            self._y = bar.close


        else:
            # set alone plot of show range
            alone_manager = self._alone_manages.get(self._plot_name)
            bar = alone_manager.get_bar(self._x)
        self._y = bar.close

        # update line and label
        self._update_line()
        self._update_label()
