from vnpy.event import Event, EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import QtCore, QtGui, QtWidgets
from vnpy.trader.ui.widget import (
    BaseCell,
    EnumCell,
    MsgCell,
    TimeCell,
    BaseMonitor
)
from vnpy.trader.utility import extract_vt_symbol, BarGenerator
from vnpy.trader.event import *
from vnpy.trader.constant import *
from vnpy.trader.database import database_manager
from ..base import (
    APP_NAME,
    EVENT_CTA_LOG,
    EVENT_CTA_STOPORDER,
    EVENT_CTA_STRATEGY
)
from ..engine import CtaEngine
from uiKline.chart.item import *
from uiKline.chart.everDataAxisWidget import *


class SendOrderText(Enum):
    SYMBOL = "合约代码"
    PRICE = "价格"
    VOLUME = "成交量"
    OFFSET = "开平"
    PRICE_TYPE = "价格类型"
    DIRECTION = "开仓方向"


class TickText(Enum):
    BID_PRICE_1 = u'买一价'
    BID_PRICE_2 = u'买二价'
    BID_PRICE_3 = u'买三价'
    BID_PRICE_4 = u'买四价'
    BID_PRICE_5 = u'买五价'
    ASK_PRICE_1 = u'卖一价'
    ASK_PRICE_2 = u'卖二价'
    ASK_PRICE_3 = u'卖三价'
    ASK_PRICE_4 = u'卖四价'
    ASK_PRICE_5 = u'卖五价'

    BID_VOLUME_1 = u'买一量'
    BID_VOLUME_2 = u'买二量'
    BID_VOLUME_3 = u'买三量'
    BID_VOLUME_4 = u'买四量'
    BID_VOLUME_5 = u'买五量'
    ASK_VOLUME_1 = u'卖一量'
    ASK_VOLUME_2 = u'卖二量'
    ASK_VOLUME_3 = u'卖三量'
    ASK_VOLUME_4 = u'卖四量'
    ASK_VOLUME_5 = u'卖五量'
    LAST = u'最新价'


class CtaManager(QtWidgets.QWidget):
    """"""

    signal_log = QtCore.pyqtSignal(Event)
    signal_strategy = QtCore.pyqtSignal(Event)

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine):
        super(CtaManager, self).__init__()

        self.main_engine = main_engine
        self.event_engine = event_engine
        self.cta_engine = main_engine.get_engine(APP_NAME)

        self.managers = {}

        self.init_ui()
        self.register_event()
        self.cta_engine.init_engine()
        self.update_class_combo()

    def init_ui(self):
        """"""
        self.setWindowTitle("CTA策略")

        # Create widgets
        self.class_combo = QtWidgets.QComboBox()

        add_button = QtWidgets.QPushButton("添加策略")
        add_button.clicked.connect(self.add_strategy)

        init_button = QtWidgets.QPushButton("全部初始化")
        init_button.clicked.connect(self.cta_engine.init_all_strategies)

        start_button = QtWidgets.QPushButton("全部启动")
        start_button.clicked.connect(self.cta_engine.start_all_strategies)

        stop_button = QtWidgets.QPushButton("全部停止")
        stop_button.clicked.connect(self.cta_engine.stop_all_strategies)

        clear_button = QtWidgets.QPushButton("清空日志")
        clear_button.clicked.connect(self.clear_log)

        self.scroll_layout = QtWidgets.QVBoxLayout()
        self.scroll_layout.addStretch()

        scroll_widget = QtWidgets.QWidget()
        scroll_widget.setLayout(self.scroll_layout)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)

        self.log_monitor = LogMonitor(self.main_engine, self.event_engine)

        self.stop_order_monitor = StopOrderMonitor(
            self.main_engine, self.event_engine
        )

        # Set layout
        hbox1 = QtWidgets.QHBoxLayout()
        hbox1.addWidget(self.class_combo)
        hbox1.addWidget(add_button)
        hbox1.addStretch()
        hbox1.addWidget(init_button)
        hbox1.addWidget(start_button)
        hbox1.addWidget(stop_button)
        hbox1.addWidget(clear_button)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(scroll_area, 0, 0, 2, 1)
        grid.addWidget(self.stop_order_monitor, 0, 1)
        grid.addWidget(self.log_monitor, 1, 1)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(grid)

        self.setLayout(vbox)

    def update_class_combo(self):
        """"""
        self.class_combo.addItems(
            self.cta_engine.get_all_strategy_class_names()
        )

    def register_event(self):
        """"""
        self.signal_strategy.connect(self.process_strategy_event)

        self.event_engine.register(
            EVENT_CTA_STRATEGY, self.signal_strategy.emit
        )

    def process_strategy_event(self, event):
        """
        Update strategy status onto its monitor.
        """
        data = event.data
        strategy_name = data["strategy_name"]

        if strategy_name in self.managers:
            manager = self.managers[strategy_name]
            manager.update_data(data)
        else:
            manager = StrategyManager(self, self.cta_engine, data)
            self.scroll_layout.insertWidget(0, manager)
            self.managers[strategy_name] = manager

    def remove_strategy(self, strategy_name):
        """"""
        manager = self.managers.pop(strategy_name)
        manager.deleteLater()

    def add_strategy(self):
        """"""
        class_name = str(self.class_combo.currentText())
        if not class_name:
            return

        parameters = self.cta_engine.get_strategy_class_parameters(class_name)
        editor = SettingEditor(parameters, class_name=class_name)
        n = editor.exec_()

        if n == editor.Accepted:
            setting = editor.get_setting()
            vt_symbol = setting.pop("vt_symbol")
            strategy_name = setting.pop("strategy_name")

            self.cta_engine.add_strategy(
                class_name, strategy_name, vt_symbol, setting
            )

    def clear_log(self):
        """"""
        self.log_monitor.setRowCount(0)

    def show(self):
        """"""
        self.showMaximized()


class SendOrderWidgets(QtWidgets.QWidget):
    signal = QtCore.pyqtSignal(type(Event))

    direction_text_class = Direction
    offset_text_class = Offset
    price_type_text_class = OrderType
    tick_text_class = TickText

    def __init__(self, ctaEngine, strategy_name, parent=None):
        super(SendOrderWidgets, self).__init__()
        self.ctaEngine = ctaEngine
        self.eventEngine = self.ctaEngine.event_engine

        self.strategy_name = strategy_name
        self.symbol = ""

        self.init()

    def init(self):
        # 设置标题
        self.setWindowTitle("%s" % self.strategy_name)

        # 固定窗口大小
        self.setFixedWidth(400)
        # self.setFixedHeight(300)

        # 左边标签
        labelSymbol = QtWidgets.QLabel()  # 合约代码
        labelDirection = QtWidgets.QLabel(SendOrderText.DIRECTION.value)  # 开仓方向
        labelOffset = QtWidgets.QLabel(SendOrderText.OFFSET.value)  # 开平
        labelPrice = QtWidgets.QLabel(SendOrderText.PRICE.value)  # 价格
        self.checkFixed = QtWidgets.QCheckBox(u'')  # 价格固定选择框
        labelVolume = QtWidgets.QLabel(SendOrderText.VOLUME.value)  # 成交量
        labelPriceType = QtWidgets.QLabel(SendOrderText.PRICE_TYPE.value)  # 市价或限价格价

        # 选择框和编辑框
        self.lineSymbol = QtWidgets.QLineEdit()
        self.lineSymbol.setReadOnly(True)
        self.comboDirection = QtWidgets.QComboBox()
        self.comboDirection.addItems([direction.value for direction in self.direction_text_class])

        self.comboOffset = QtWidgets.QComboBox()
        self.comboOffset.addItems([offset.value for offset in self.offset_text_class])

        validator = QtGui.QDoubleValidator()  # 浮点编辑器
        validator.setBottom(0)  # 显然价格 成交量没有负数

        self.linePrice = QtWidgets.QLineEdit()
        self.linePrice.setValidator(validator)

        self.lineVolume = QtWidgets.QLineEdit()
        self.lineVolume.setValidator(validator)

        self.comboPriceType = QtWidgets.QComboBox()
        self.comboPriceType.addItems([price_type.value for price_type in self.price_type_text_class])

        grid_left = QtWidgets.QGridLayout()
        grid_left.addWidget(labelSymbol, 0, 0)
        grid_left.addWidget(labelDirection, 1, 0)
        grid_left.addWidget(labelOffset, 2, 0)
        grid_left.addWidget(labelPrice, 3, 0)
        grid_left.addWidget(labelVolume, 4, 0)
        grid_left.addWidget(labelPriceType, 5, 0)

        grid_left.addWidget(self.lineSymbol, 0, 1, 1, -1)
        grid_left.addWidget(self.comboDirection, 1, 1, 1, -1)
        grid_left.addWidget(self.comboOffset, 2, 1, 1, -1)
        grid_left.addWidget(self.checkFixed, 3, 1)
        grid_left.addWidget(self.linePrice, 3, 2, 1, -1)
        grid_left.addWidget(self.lineVolume, 4, 2, 1, -1)
        grid_left.addWidget(self.comboPriceType, 5, 1, 1, -1)

        # 右边行情
        labelBid1 = QtWidgets.QLabel(self.tick_text_class.BID_PRICE_1.value)
        labelBid2 = QtWidgets.QLabel(self.tick_text_class.BID_PRICE_2.value)
        labelBid3 = QtWidgets.QLabel(self.tick_text_class.BID_PRICE_3.value)

        labelAsk1 = QtWidgets.QLabel(self.tick_text_class.ASK_PRICE_1.value)
        labelAsk2 = QtWidgets.QLabel(self.tick_text_class.ASK_PRICE_2.value)
        labelAsk3 = QtWidgets.QLabel(self.tick_text_class.ASK_PRICE_3.value)

        self.labelBidPrice1 = QtWidgets.QLabel()
        self.labelBidPrice2 = QtWidgets.QLabel()
        self.labelBidPrice3 = QtWidgets.QLabel()
        self.labelBidVolume1 = QtWidgets.QLabel()
        self.labelBidVolume2 = QtWidgets.QLabel()
        self.labelBidVolume3 = QtWidgets.QLabel()

        self.labelAskPrice1 = QtWidgets.QLabel()
        self.labelAskPrice2 = QtWidgets.QLabel()
        self.labelAskPrice3 = QtWidgets.QLabel()
        self.labelAskVolume1 = QtWidgets.QLabel()
        self.labelAskVolume2 = QtWidgets.QLabel()
        self.labelAskVolume3 = QtWidgets.QLabel()

        labelLast = QtWidgets.QLabel(self.tick_text_class.LAST.value)
        self.labelLastPrice = QtWidgets.QLabel()

        grid_right = QtWidgets.QGridLayout()

        grid_right.addWidget(labelAsk3, 0, 0)
        grid_right.addWidget(labelAsk2, 1, 0)
        grid_right.addWidget(labelAsk1, 2, 0)
        grid_right.addWidget(labelLast, 3, 0)
        grid_right.addWidget(labelBid1, 4, 0)
        grid_right.addWidget(labelBid2, 5, 0)
        grid_right.addWidget(labelBid3, 6, 0)

        grid_right.addWidget(self.labelAskPrice3, 0, 1)
        grid_right.addWidget(self.labelAskPrice2, 1, 1)
        grid_right.addWidget(self.labelAskPrice1, 2, 1)
        grid_right.addWidget(self.labelLastPrice, 3, 1)
        grid_right.addWidget(self.labelBidPrice1, 4, 1)
        grid_right.addWidget(self.labelBidPrice2, 5, 1)
        grid_right.addWidget(self.labelBidPrice3, 6, 1)

        grid_right.addWidget(self.labelAskVolume3, 0, 2)
        grid_right.addWidget(self.labelAskVolume2, 1, 2)
        grid_right.addWidget(self.labelAskVolume1, 2, 2)
        grid_right.addWidget(self.labelBidVolume1, 4, 2)
        grid_right.addWidget(self.labelBidVolume2, 5, 2)
        grid_right.addWidget(self.labelBidVolume3, 6, 2)

        # 布局整合
        hbox = QtWidgets.QHBoxLayout()
        hbox.addLayout(grid_left)
        hbox.addLayout(grid_right)

        self.send_botton = QtWidgets.QPushButton("发单")

        vbox = QtWidgets.QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.send_botton)
        vbox.addStretch()

        self.setLayout(vbox)

        self.send_botton.clicked.connect(self.send_order)

    # ---------------------------------
    def send_order(self):
        strategy = self.ctaEngine.strategies.get(self.strategy_name)
        if not strategy:
            return

        # 获取价格
        price = self.linePrice.text()
        if not price:
            return
        price = float(price)

        # 获取数量
        volumeText = self.lineVolume.text()
        if not volumeText:
            return

        if '.' in volumeText:
            volume = float(volumeText)
        else:
            volume = int(volumeText)

        # 方向判断
        direction = self.comboDirection.currentText()
        offset = self.comboOffset.currentText()

        if direction == self.direction_text_class.LONG.value and offset == self.offset_text_class.OPEN.value:
            strategy.buy(price, volume)
        elif direction == self.direction_text_class.SHORT.value and offset == self.offset_text_class.CLOSE.value:
            strategy.sell(price, volume)
        elif direction == self.direction_text_class.SHORT.value and offset == self.offset_text_class.OPEN.value:
            strategy.short(price, volume)
        if direction == self.direction_text_class.LONG.value and offset == self.offset_text_class.CLOSE.value:
            strategy.cover(price, volume)

    # ---------------------------------
    def start(self):
        self.set_data()
        self.registerEvent()
        self.show()

    def set_data(self):
        strategy = self.ctaEngine.strategies.get(self.strategy_name)
        if not strategy:
            return

        self.symbol = strategy.vt_symbol
        self.lineSymbol.setText(self.symbol)

    def registerEvent(self):
        """事件注册"""
        if self.ctaEngine:
            self.signal.connect(self.update_tick)
            self.eventEngine.register(EVENT_TICK, self.signal.emit)

    def closeEvent(self, QCloseEvent):
        if self.ctaEngine:
            self.eventEngine.unregister(EVENT_TICK, self.signal.emit)
        super(SendOrderWidgets, self).closeEvent(QCloseEvent)

    def update_tick(self, event):
        """market tick update"""

        tick = event.data
        if tick.vt_symbol != self.symbol:
            return
        if not self.checkFixed.isChecked():
            self.linePrice.setText(str(tick.last_price))

        self.labelBidPrice1.setText(str(tick.bid_price_1))
        self.labelAskPrice1.setText(str(tick.ask_price_1))
        self.labelBidVolume1.setText(str(tick.bid_volume_1))
        self.labelAskVolume1.setText(str(tick.ask_volume_1))

        if tick.bid_price2:
            self.labelBidPrice2.setText(str(tick.bid_price_2))
            self.labelBidPrice3.setText(str(tick.bid_price_3))

            self.labelAskPrice2.setText(str(tick.ask_price2))
            self.labelAskPrice3.setText(str(tick.ask_price3))

            self.labelBidVolume2.setText(str(tick.bid_volume_2))
            self.labelBidVolume3.setText(str(tick.bid_volume_3))

            self.labelAskVolume2.setText(str(tick.ask_volume_2))
            self.labelAskVolume3.setText(str(tick.ask_volume_3))
        self.labelLastPrice.setText(str(tick.last_price))


class PlotWidgets(EverDataWidget):
    signal = QtCore.pyqtSignal(type(Event))

    def __init__(self, cta_engine, data: dict, parent=None):
        super(PlotWidgets, self).__init__(parent)
        self.cta_engine = cta_engine
        self._data = data
        self.vt_symbol = self._data.get("vt_symbol")

        self.strategy_name = self._data.get("strategy_name")
        self.generate_bar = BarGenerator(self.on_bar)
        self.bar_list = []
        self.init_ui()
        self.close()

    def init_ui(self):
        self.setWindowTtile(f"{self.strategy_name}")

        self.add_plot("candle_item", hide_x_axis=True)
        self.add_plot("volume_item")

        self.add_item(CandleItem, "candle", "candle")
        self.add_item(VolumeItem, "volume", "volume")
        self.add_cursor()

    def load_all_bars(self):
        if self.vt_symbol and not self.bar_list:
            symbol, exchange = extract_vt_symbol(self.vt_symbol)
            exchange = Exchange(exchange)
            bars = mongo_manager.load_all_the_bars_data(symbol, exchange)
            self.bar_list.extend(bars)

    def init_data(self):
        self.load_all_bars()
        if self.bar_list:
            self.update_history(self.bar_list)

    def start_plot(self):

        self.init_data()
        self.register()
        self.show()

    def stop_plot(self):
        self.unregister()
        self.bar_list.clear()

    def register(self):
        event_engine = self.cta_engine.event_engine
        event_engine.register(EVENT_TICK, self.on_tick)

    def unregister(self):
        event_engine = self.cta_engine.event_engine
        event_engine.unregister(EVENT_TICK, self.on_tick)

    def on_tick(self, event):
        tick = event.data
        if not tick.vt_symbol == self.vt_symbol:
            return
        self.generate_bar.update_tick(tick)

    def close(self):
        self.start_plot()
        super(PlotWidgets, self).close()

    def on_bar(self, bar):
        self.update_bar(bar)


class StrategyManager(QtWidgets.QFrame):
    """
    Manager for a strategy
    """

    def __init__(
            self, cta_manager: CtaManager, cta_engine: CtaEngine, data: dict
    ):
        """"""
        super(StrategyManager, self).__init__()

        self.cta_manager = cta_manager
        self.cta_engine = cta_engine

        self.strategy_name = data["strategy_name"]
        self._data = data

        self.plot_widget = PlotWidgets(cta_engine, data)
        self.init_ui()

    def init_ui(self):
        """"""
        self.setFixedHeight(300)
        self.setFrameShape(self.Box)
        self.setLineWidth(1)

        init_button = QtWidgets.QPushButton("初始化")
        init_button.clicked.connect(self.init_strategy)

        start_button = QtWidgets.QPushButton("启动")
        start_button.clicked.connect(self.start_strategy)

        stop_button = QtWidgets.QPushButton("停止")
        stop_button.clicked.connect(self.stop_strategy)

        edit_button = QtWidgets.QPushButton("编辑")
        edit_button.clicked.connect(self.edit_strategy)

        remove_button = QtWidgets.QPushButton("移除")
        remove_button.clicked.connect(self.remove_strategy)

        # add manual operator send order function
        send_order_button = QtWidgets.QPushButton("发单")
        send_order_button.clicked.conect(self.send_order)

        paint_chart_button = QtWidgets.QPushButton("画图")
        paint_chart_button.clicked.connect(self.plot)
        strategy_name = self._data["strategy_name"]
        vt_symbol = self._data["vt_symbol"]
        class_name = self._data["class_name"]
        author = self._data["author"]

        label_text = (
            f"{strategy_name}  -  {vt_symbol}  ({class_name} by {author})"
        )
        label = QtWidgets.QLabel(label_text)
        label.setAlignment(QtCore.Qt.AlignCenter)

        self.parameters_monitor = DataMonitor(self._data["parameters"])
        self.variables_monitor = DataMonitor(self._data["variables"])

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(init_button)
        hbox.addWidget(start_button)
        hbox.addWidget(stop_button)
        hbox.addWidget(edit_button)
        hbox.addWidget(remove_button)
        # add a new button
        hbox.addWidget(send_order_button)
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(label)
        vbox.addLayout(hbox)
        vbox.addWidget(self.parameters_monitor)
        vbox.addWidget(self.variables_monitor)
        self.setLayout(vbox)

    def update_data(self, data: dict):
        """"""
        self._data = data

        self.parameters_monitor.update_data(data["parameters"])
        self.variables_monitor.update_data(data["variables"])

    def init_strategy(self):
        """"""
        self.cta_engine.init_strategy(self.strategy_name)

    def start_strategy(self):
        """"""
        self.cta_engine.start_strategy(self.strategy_name)

    def stop_strategy(self):
        """"""
        self.cta_engine.stop_strategy(self.strategy_name)

    def edit_strategy(self):
        """"""
        strategy_name = self._data["strategy_name"]

        parameters = self.cta_engine.get_strategy_parameters(strategy_name)
        editor = SettingEditor(parameters, strategy_name=strategy_name)
        n = editor.exec_()

        if n == editor.Accepted:
            setting = editor.get_setting()
            self.cta_engine.edit_strategy(strategy_name, setting)

    def send_order(self):
        send_order = SendOrderWidgets(self.cta_engine, self.strategy_name)
        send_order.start()

    def plot(self):
        self.plot_widget.start_plot()

    def remove_strategy(self):
        """"""
        result = self.cta_engine.remove_strategy(self.strategy_name)

        # Only remove strategy gui manager if it has been removed from engine
        if result:
            self.cta_manager.remove_strategy(self.strategy_name)


class DataMonitor(QtWidgets.QTableWidget):
    """
    Table monitor for parameters and variables.
    """

    def __init__(self, data: dict):
        """"""
        super(DataMonitor, self).__init__()

        self._data = data
        self.cells = {}

        self.init_ui()

    def init_ui(self):
        """"""
        labels = list(self._data.keys())
        self.setColumnCount(len(labels))
        self.setHorizontalHeaderLabels(labels)

        self.setRowCount(1)
        self.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(self.NoEditTriggers)

        for column, name in enumerate(self._data.keys()):
            value = self._data[name]

            cell = QtWidgets.QTableWidgetItem(str(value))
            cell.setTextAlignment(QtCore.Qt.AlignCenter)

            self.setItem(0, column, cell)
            self.cells[name] = cell

    def update_data(self, data: dict):
        """"""
        for name, value in data.items():
            cell = self.cells[name]
            cell.setText(str(value))


class StopOrderMonitor(BaseMonitor):
    """
    Monitor for local stop order.
    """

    event_type = EVENT_CTA_STOPORDER
    data_key = "stop_orderid"
    sorting = True

    headers = {
        "stop_orderid": {
            "display": "停止委托号",
            "cell": BaseCell,
            "update": False,
        },
        "vt_orderids": {"display": "限价委托号", "cell": BaseCell, "update": True},
        "vt_symbol": {"display": "本地代码", "cell": BaseCell, "update": False},
        "direction": {"display": "方向", "cell": EnumCell, "update": False},
        "offset": {"display": "开平", "cell": EnumCell, "update": False},
        "price": {"display": "价格", "cell": BaseCell, "update": False},
        "volume": {"display": "数量", "cell": BaseCell, "update": False},
        "status": {"display": "状态", "cell": EnumCell, "update": True},
        "lock": {"display": "锁仓", "cell": BaseCell, "update": False},
        "strategy_name": {"display": "策略名", "cell": BaseCell, "update": False},
    }


class LogMonitor(BaseMonitor):
    """
    Monitor for log data.
    """

    event_type = EVENT_CTA_LOG
    data_key = ""
    sorting = False

    headers = {
        "time": {"display": "时间", "cell": TimeCell, "update": False},
        "msg": {"display": "信息", "cell": MsgCell, "update": False},
    }

    def init_ui(self):
        """
        Stretch last column.
        """
        super(LogMonitor, self).init_ui()

        self.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch
        )

    def insert_new_row(self, data):
        """
        Insert a new row at the top of table.
        """
        super(LogMonitor, self).insert_new_row(data)
        self.resizeRowToContents(0)


class SettingEditor(QtWidgets.QDialog):
    """
    For creating new strategy and editing strategy parameters.
    """

    def __init__(
            self, parameters: dict, strategy_name: str = "", class_name: str = ""
    ):
        """"""
        super(SettingEditor, self).__init__()

        self.parameters = parameters
        self.strategy_name = strategy_name
        self.class_name = class_name

        self.edits = {}

        self.init_ui()

    def init_ui(self):
        """"""
        form = QtWidgets.QFormLayout()

        # Add vt_symbol and name edit if add new strategy
        if self.class_name:
            self.setWindowTitle(f"添加策略：{self.class_name}")
            button_text = "添加"
            parameters = {"strategy_name": "", "vt_symbol": ""}
            parameters.update(self.parameters)
        else:
            self.setWindowTitle(f"参数编辑：{self.strategy_name}")
            button_text = "确定"
            parameters = self.parameters

        for name, value in parameters.items():
            type_ = type(value)

            edit = QtWidgets.QLineEdit(str(value))
            if type_ is int:
                validator = QtGui.QIntValidator()
                edit.setValidator(validator)
            elif type_ is float:
                validator = QtGui.QDoubleValidator()
                edit.setValidator(validator)

            form.addRow(f"{name} {type_}", edit)

            self.edits[name] = (edit, type_)

        button = QtWidgets.QPushButton(button_text)
        button.clicked.connect(self.accept)
        form.addRow(button)

        self.setLayout(form)

    def get_setting(self):
        """"""
        setting = {}

        if self.class_name:
            setting["class_name"] = self.class_name

        for name, tp in self.edits.items():
            edit, type_ = tp
            value_text = edit.text()

            if type_ == bool:
                if value_text == "True":
                    value = True
                else:
                    value = False
            else:
                value = type_(value_text)

            setting[name] = value

        return setting
