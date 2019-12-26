# -*- coding: utf-8 -*-
from chart.widget import *
from chart.item import *
from chart.item_for_average import *
from chart.manager import *
from chart.everDataAxisWidget import *
from time import sleep

# 默认空值
EMPTY_STRING = ''
EMPTY_UNICODE = u''
EMPTY_INT = 0
EMPTY_FLOAT = 0.0


class VtBarData():
    """K线数据"""

    # ----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        super(VtBarData, self).__init__()

        self.vtSymbol = EMPTY_STRING  # vt系统代码
        self.symbol = EMPTY_STRING  # 代码
        self.exchange = EMPTY_STRING  # 交易所

        self.open = EMPTY_FLOAT  # OHLC
        self.high = EMPTY_FLOAT
        self.low = EMPTY_FLOAT
        self.close = EMPTY_FLOAT

        self.date = EMPTY_STRING  # bar开始的时间，日期
        self.time = EMPTY_STRING  # 时间
        self.datetime = None  # python的datetime时间对象

        self.volume = EMPTY_INT  # 成交量
        self.openInterest = EMPTY_INT  # 持仓量
        self.interval = EMPTY_UNICODE  # K线周期

        self.dataSource = EMPTY_STRING  # newfunc 数据来源


if __name__ == '__main__':
    import random
    import datetime as dt
    import pandas as pd
    # from vnpy.trader.vtObject import VtBarData
    from PyQt5.QtWidgets import QApplication
    import sys

    # 创建barlist
    data = pd.read_csv("RB9999.csv", index_col="datetime")
    sig_data = pd.read_csv("signal.csv", index_col="datetime")

    list_bar = []
    for index, row in data.iterrows():
        bar = VtBarData()
        bar.datetime = dt.datetime.strptime(index, "%Y/%m/%d")
        bar.low = float(row.low)
        bar.high = float(row.high)
        bar.open = float(row.open)
        bar.close = float(row.close)
        bar.volume = int(row.volume)
        list_bar.append(bar)

    # 初始化
    app = QApplication(sys.argv)
    # widget = ChartWidget()
    widget = EverDataWidget()

    # 添加子图plot
    widget.add_plot("candle", hide_x_axis=True)
    widget.add_plot("volume", maximum_height=200)

    # 添加图层item
    widget.add_item(CandleItem, "candle", "candle")  # 添加K线
    # widget.add_item(LineItem, "k_line", "candle")       # 添加静态线
    widget.add_item(AverageItem, "average", "candle")  # 添加均线图层
    widget.add_item(ArrowItem, "sig_arrow", "candle")  # 添加箭头
    widget.add_item(VolumeItem, "volume", "volume")  # 添加成交量
    widget.add_cursor()  # 添加十字光标

    # 添加独立x轴candle plot
    widget.add_plot("test_candle_plot", maximum_height=200, common=False)
    # alone x-axis volume plot
    widget.add_plot("test_volume_plot", maximum_height=200, common=False)

    # add Item in to alone x-axis plot
    widget.add_item(CandleItem, "test_candle", "test_candle_plot")
    widget.add_item(LineItem, "test_k_line", "test_candle_plot")
    widget.add_item(VolumeItem, "volume", "test_volume_plot")
    widget.add_cursor()

    widget.update_history(list_bar[:300])
    # alone manager add to alone plot with assign name of plot
    widget.update_history(list_bar[:300], alone=True, alone_plot_name="test_candle_plot")
    widget.update_history(list_bar[:300], alone=True, alone_plot_name="test_volume_plot")
    widget.update_bar_signal(sig_data)  # 添加信号箭头数据（决定在哪画）

    arrow_plot = widget._items.get("sig_arrow")  # 获取箭头Item
    arrow_plot.update_history()  # 刷新箭头图层bar 开始画箭头 不刷新不会有箭头图

    widget.show()

    common_plot_index = 300
    alone_plot_index = 300


    def update_common_data():
        global common_plot_index
        widget.update_bar(list_bar[common_plot_index])
        common_plot_index += 1


    def update_alone_plot_data():
        global alone_plot_index
        widget.update_bar(list_bar[alone_plot_index], True, "test_candle_plot")
        alone_plot_index += 1


    # all axis plot update bar
    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update_common_data)
    timer.start(1000)

    # alone axis plot update bar
    alone_timer = pg.QtCore.QTimer()
    alone_timer.timeout.connect(update_alone_plot_data)
    alone_timer.start(100)

    sys.exit(app.exec_())
