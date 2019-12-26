# encoding: UTF-8
from __future__ import division
import re,math,webbrowser,os,traceback,linecache,time,pytz,pymongo,json,multiprocessing,decimal
from threading import Thread

import pandas as pd
import numpy as np
import talib as ta
import datetime as dt

# import matplotlib.pyplot as plt
from pyecharts import *

import statsmodels.api as sm
from collections import OrderedDict
from copy import deepcopy
from functools import wraps

pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',100)
pd.set_option('display.width', 1000)
pd.set_option('expand_frame_repr', False)  # 当列太多时不自动换行
pd.set_option('display.float_format', lambda x: '%.2f' % x)  # 设置表的长度、宽度 & 不采用科学计数法

"""一、基础工具"""
# [python2] pickle化实例方法
def py2_pickle():
    """
    python2.7不支持实例方法pickle化(会出现PicklingError)，因此需要使用copy_reg将MethodType注册为可序列化
    更多方法可以见 https://blog.tankywoo.com/2015/09/06/cant-pickle-instancemethod.html

    :return:
    """
    import copy_reg
    import types
    def _reduce_method(m):
         if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
         else:
             return getattr, (m.im_self, m.im_func.func_name)
    copy_reg.pickle(types.MethodType, _reduce_method)

# [python3] 调整import导包顺序
def python3_add_import_path(import_path):
    """Python3导包顺序设置

    eg.
    import os
    path_dir = os.path.dirname(__file__)
    python3_add_import_path(path_dir)
    """
    import sys
    system_import_path = sys.path       # 系统导入路径
    sys.path.insert(0, import_path)     # remark system_import_path[0]是运行文件的路径，应为最优先，所以要插入到第1个处

# [python3] list化
def map2(func, *ll):
    return list(map(func, *ll))

def range2(*ll):
    return list(range(*ll))

def zip2(*ll):
    return list(zip(*ll))


# [print] 输出unicode中文(python2)
def printChinese(s):
    print(str(s).decode('unicode-escape'))

# [print] 完整输出data
def printData(data, max_rows=6000):
    pd.set_option('display.max_rows',max_rows)
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    print(data)
    pd.set_option('display.max_rows',50)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

# [print] 输出dict的每项数据
def printDict(d):
    for k,v in d.items():
        print('"{}": {}'.format(k, v))


# [file] 获得path下的每一个文件的文件夹名、文件名、文件路径
def getDirFile(path):
    l = []
    for root,dirs,files in os.walk(path):
        for f in files:
            dirname = os.path.split(root)[1]       # 文件夹名
            filename = f                        # 文件名
            filepath = os.path.join(root, f)    # 文件路径
            l.append([dirname,filename,filepath])
    return l

# [file] 重命名
def rename(path):
    pathlist = getDirFile(path)
    for dbname, filename, filepath in pathlist:
        # notice 在这里写重命名规则
        newfilepath = filepath + '.H5'

        # 重命名之前最好先断点看一下
        print(filepath, newfilepath)
        # os.rename(filepath,newfilepath)

# [file] 安全路径
def safepath(path):
    """
    功能：
    1.检查路径的所在的文件夹是否存在，不在则创建文件夹
    2.合成路径

    args:
    传入单个路径str： safepath('G://123/XBTCUSD.H5')，
    传入多个路径list：safepath(['G://123','XBTCUSD.H5'])
    """
    # notice 文件名里不能包含'/''\'。1.windows不支持这两个字符做文件名；2.如果路径里面包含'/''\'会被拆为文件夹名
    if isinstance(path, str):
        dirpath, filename = os.path.split(path)
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        return path
    elif isinstance(path, list):
        if '/' in path[-1] or '\\' in path[-1]:
            path[-1] = path[-1].replace('/', '%').replace('\\', '%')
            print(r"Warning: Filename can't use '\' or '/', it will replace to: %s" %(path[-1]))
        path0 = os.path.join(*path)

        dirpath, filename = os.path.split(path0)
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        return path0


# [numeral] 浮点数运算decimal
def dec(x):
    """
    decimal传入int或者str，但不能传入float。
    如果对decimal传入float会不准确，因此需要把float转换成str再传入。
    eg.
    Decimal(1.45)                       # Decimal('1.4499999999999999555910790149937383830547332763671875')
    Decimal(str(1.45'))                 # Decimal('1.45')
    """
    y = decimal.Decimal(str(x))
    return y

# [numeral] 判断x是否为nan（可以传入单个值，也可以传入非纯数序列）
def isnan(x):
    # 如果传入的是
    try:
        y = np.isnan(x)
    except TypeError:
        if not isinstance(x, str):
            y = [True if i!=i else False for i in x]
        else:
            y = False
    return y


# [list] 去除list中的nan
def filter_nan(ll):
    return list(filter(lambda x: x if not math.isnan(x) else None, ll))

# [list] 替换list中的nan
def replace_nan(ll, replacevalue):
    return list(map(lambda x: x if not math.isnan(x) else replacevalue, ll))

# [list] 比较listXY的不同
def list_compare(x,y):
    XYSame = [i for i in x if i in y]
    XOnly = [i for i in x if i not in y]
    YOnly = [i for i in y if i not in x]
    return XOnly, YOnly, XYSame

# [list] list/np.array的shift(np.roll会把最后的值移到第一位)
def shift(ll, n=1):
    if isinstance(ll, np.ndarray):
        return pd.Series(ll).shift(n).values
    elif isinstance(ll, list):
        return pd.Series(ll).shift(n).tolist()

# [list] list去重
def list_dropDuplicates(l):
    l2 = list(set(l))
    l2.sort(key=l.index)
    return l2


# [numpy] 替换原来numpy的ndarray，增加update和dropna方法
class ndarray(np.ndarray):
    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        return obj

    def update(self, newValue):
        self[:-1] = self[1:]
        self[-1] = newValue

    def dropna(self):
        # return filter(lambda x: x if x is not None else None, self)   # filter会把0也过滤掉
        return [i for i in self if i is not None]   # 列表生成式不会把0过滤掉

# [numpy] 更新np.ndarray
def updateArray(array, newValue):
    array[:-1] = array[1:]
    array[-1] = newValue


# [pandas] 取df1、df2的交集、并集
def dataframe_intersection(dfa, dfb, on, how='inner', resetindex=True):
    df1,df2 = dfa.copy(),dfb.copy()
    df1.columns = map(lambda x: x+'_1' if x!=on else x, df1.columns) # 对除了on之外的columns添加后缀。PS.不要直接用pd.merge(suffixes=['_x','_y'])，因为suffixes只对重复值添加后缀
    df2.columns = map(lambda x: x+'_2' if x!=on else x, df2.columns)
    if resetindex == False:
        df1['index_1'],df2['index_2'] = df1.index,df2.index
    df = pd.merge(left=df1, right=df2, on=on, how=how)
    df3,df4 = df.ix[:,df1.columns],df.ix[:,df2.columns]
    if resetindex == False:
        df3.set_index('index_1', inplace=True),df4.set_index('index_2', inplace=True)
    df3.columns,df4.columns = dfa.columns,dfb.columns
    return df3,df4

# [pandas] 给DataFrame添加新列
def dataframe_add_columns(data, new_data_dict=None, new_columnsName_list=None):
    """

    :param data: pd.DataFrame
    :param new_data_dict: dict。eg. {'lastPrice': 0, 'openInterest': np.nan, 'symbol': ''}
    :param new_columnsName_list: list。 eg. ['lastPrice', 'openInterest', 'symbol']
    :return:

    eg.
    dataframe_add_columns(df, new_columnsName_list=['datetime', 'lastPrice', 'openInterest', 'lastVolume', 'volume', 'bidPrice1', 'askPrice1'])
    dataframe_add_columns(df, new_data_dict={'lastPrice': 0, 'openInterest': np.nan, 'symbol': ''})
    """
    if new_data_dict:
        new = pd.DataFrame(new_data_dict, index=range(len(data)))
    else:
        new = pd.DataFrame(columns=new_columnsName_list)
    return pd.concat([data, new], axis=1)

# [pandas] DataFrame内存占用优化函数
def dataframe_reduceMem(data):
    """
    把DataFrame的占用内存大的变量类型，转换为占用内存更小的变量类型。

    具体方法：
    1.循环每列
    2.判断是否该列类型为numeric
    3.判断是否该列类型为int
    4.找到最小最大值
    5.找到一个最节省内存的datatype去fit这一列

    原文：https://zhuanlan.zhihu.com/p/68092069
    据评论反映，对datatime的处理好像会有点问题？需要特别注意。

    eg.
    df = pd.read_csv('rb0000_1min.csv')
    data, NAlist = dataframe_reduceMem(df)
    print(data, NAlist)
    """
    start_mem_usg = data.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in.
    for col in data.columns:
        if data[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",data[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = data[col].max()
            mn = data[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(data[col]).all():
                NAlist.append(col)
                data[col].fillna(mn-1,inplace=True)

            # test if column can be converted to an integer
            asint = data[col].fillna(0).astype(np.int64)
            result = (data[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True


            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        data[col] = data[col].astype(np.uint8)
                    elif mx < 65535:
                        data[col] = data[col].astype(np.uint16)
                    elif mx < 4294967295:
                        data[col] = data[col].astype(np.uint32)
                    else:
                        data[col] = data[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        data[col] = data[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        data[col] = data[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        data[col] = data[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        data[col] = data[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                data[col] = data[col].astype(np.float32)

            # Print new column type
            print("dtype after: ",data[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = data.memory_usage().sum() / 1024**2
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return data, NAlist


# [datetime] 时区转换
def turnTimezone_utc8(datetime0, timezone='UTC'):
    timezone_china = pytz.timezone("Asia/Shanghai") # 设置中国时区
    datetime1 = datetime0.replace(tzinfo=pytz.timezone(timezone)).astimezone(timezone_china)  # datetime0添加时区为timezone，然后转换为中国时区
    datetime1 = datetime1.replace(tzinfo=None)  # offset-aware（带时区的datetime格式） 转换为 offset-naive（不带时区的datetime格式）
    return datetime1


# [HDF] 保存hdf（压缩全等值列到'compress'列）
def save_hdf(savedata, savepath, compress=False, dsname=None):
    savepath_file = safepath(savepath)
    dsname = dsname if dsname else 'data'

    # 如果不压缩，直接保存；如果压缩，则取出全相等的列，保存为data['compress'][0]=dict(col:values)  # remark 因为是hdf5，且保存格式是dict，所以源数据的格式都会被保存
    if compress:
        savedata = deepcopy(savedata)   # deepcopy防止修改存入数据的内存

        d = {}
        for col in savedata.columns:
            if (savedata[col] == savedata[col][0]).all():
                d[col] = savedata[col][0]
        savedata.drop(labels=d.keys(), axis=1, inplace=True)
        savedata['compress'] = None
        savedata['compress'][0] = d

    h5 = pd.HDFStore(savepath_file, 'a', complevel=4, complib='blosc')
    h5[dsname] = savedata
    h5.close()

    # 输出结果
    print('compress: %s; savefile: '%(compress) +savepath_file)

# [HDF] 读取hdf
def read_hdf(path, dsname=None):
    """
    :param path:
    :param dsname:
    :return:
    """
    h5 = pd.HDFStore(path, 'r', complevel=4, complib='blosc')
    dsname = dsname if dsname else h5.keys()[0]
    data = h5[dsname]

    # 展开压缩列'compress'列并恢复原状
    if 'compress' in data.columns:
        d = data['compress'][0]
        newdata = pd.DataFrame(d, index=range(len(data)))
        data = data.join(newdata)
        data.drop('compress', axis=1, inplace=True)
    h5.close()
    return data



"""二、通用工具"""
# [绘图] pyecharts
class plotPyecharts(object):
    """
    subgraph：子图
    layer：图层
    Overlap：一幅子图里含有多个图层
    Grid：一幅网页图里含有多个子图

    from vnpy.utor.utFunction import *
    data = pd.read_csv(r'G:\IF888_1D.CSV')
    data['datetime'] = data['datetime'].map(str)  # notice datetime必须是str格式
    pic = plotPyecharts()
    pic.plot(1, data['datetime'], data[['open','close','low','high']], legend='kline', type=Kline)
    pic.plot(2, data['datetime'], data['open'], legend='open', type=Line)
    pic.plot(len(pic.pool)+1, data['datetime'], data['capital'], legend='capital', type=Line)
    pic.render(page_title='KLine')
    """
    # notice 传入的数据必须是str或者float/int格式
    def __init__(self):
        self.pool = OrderedDict()   # 图层池 {子图1: {图层编号1:图层实例1, 图层编号2:图层实例2, ...}, 子图2: {图层编号1:图层实例1, 图层编号2:图层实例2, ...}}
        self.layerID = 0            # 初始图层编号

        self.initsetting = {'width':1600, 'height':900}
        self.layersetting = {'is_datazoom_show':True, 'datazoom_type':'both',               # 使用滚轮缩放
                             'datazoom_range':[0,100],                                      # 初始图像缩放范围
                            'tooltip_axispointer_type':'cross', 'tooltip_tragger':'axis',   # 使用十字线触发提示框
                            'yaxis_min':'dataMin','yaxis_max':'dataMax',                    # 图像自适应最小值
                             'datazoom_xaxis_index':[],                                     # 多图缩放名单（逐步添加）
                            }

    # 1.传入图层数据：把传入的图层参数转换为dict形式存放在缓存.
    def plot(self, subgraph, x, y, legend=None, type=None, **kwargs):
        """
        :param subgraph: 图层
        :param x: 横坐标。如果x=None，则x为range(y)。
        :param y: 数据
        :param legend: 图例名，str
        :param type: Line/Kline/...
        :param kwargs: 其他图层设定
        """
        if not type:
            type = Line

        # 图层缓存
        self.layerID += 1                           # 图层编号
        if subgraph not in list(self.pool.keys()):  # 在图层池创建子图缓存
            self.pool[subgraph] = OrderedDict()

        # 图层设定
        legend = str(self.layerID) if legend is None else legend                                                        # 设置图例名
        self.layersetting['datazoom_xaxis_index'].append(subgraph-1)                                                    # 在多图缩放名单添加子图
        self.pool[subgraph][self.layerID] = dict(subgraph=subgraph, x=x, y=y, legend=legend, type=type, kwargs=kwargs)  # 把所有参数存入图层池（以dict形式）
        return self

    # 2.建立图层实例layer：读取缓存的图层参数，建立图层实例
    def addLayer(self, layerdict):
        # 数据转换
        if layerdict['x'] is None:                                                  # 如果x=None, 则把x转为y的索引序列
            layerdict['x'] = range2(len(layerdict['y']))
        if layerdict['type']==Kline and isinstance(layerdict['y'], pd.DataFrame):   # 自动把KLine的DataFrame转换为list
            layerdict['y'] = layerdict['y'].values.tolist()

        layer = layerdict['type'](**self.initsetting)                               # 建立图层实例
        layersetting = dict(self.layersetting,**layerdict['kwargs'])                # 设入图层设定
        if len(layerdict['y'])>20000:                                               # 设定缩放范围（默认在30000个数据点以内）
            layersetting['datazoom_range'] = [0,20000/len(layerdict['y'])*100]
        layer.add(name=layerdict['legend'], x_axis=layerdict['x'], y_axis=layerdict['y'], **layersetting)
        return layer

    # 3.合成Grid图：把各个图层实例合成到Grid图
    def addGrid(self, piclist, gridratio=None):
        gridratio = [(10.0 / len(piclist))]*len(piclist) if gridratio is None else gridratio
        gridnum = 0
        grid = Grid(**self.initsetting)
        for picgrid in piclist:
            grid_top = sum(gridratio[:gridnum]) * 10
            grid_bottom = sum(gridratio[gridnum+1:]) * 10
            gridnum += 1

            if grid_top == 0:
                grid.add(picgrid, grid_bottom='%s%%'%(grid_bottom))
            elif grid_bottom == 0:
                grid.add(picgrid, grid_top='%s%%'%(grid_top))
            else:
                grid.add(picgrid, grid_top='%s%%'%(grid_top), grid_bottom='%s%%'%(grid_bottom))
        return grid

    # 4.执行合成并保存（实际执行函数，前面1,2,3部分在这部分执行）
    def render(self, page_title='Echarts', gridratio=None, open_now=True, dirpath=None):
        """
        :param page_title: html文件名 & 网页标签名
        :param gridratio:  子图比例
        :param open_now:   是否立刻打开html
        :return:
        """
        # todo 各子图比例可以不用在render里统一gridratio，而是在创建子图时加入参数ratio（ratio可以等于10~100），然后在render时把各个子图的ratio加总除权，得到各子图的比例。
        # 初始化
        subgraphpool = OrderedDict()                                        # 缓存池
        self.initsetting['page_title'] = page_title                         # html文件名 & 网页标签名
        every_legend_pos = (self.initsetting['width']/(self.layerID+1))# 图例位置 = (图片宽度/总图层数) * 图层编码layerID

        # Overlap图层合成子图：如果子图图层数大于1，合成为Overlap并保存到subgraphpool，否则直接保存到subgraphpool
        for subgraph in list(self.pool.keys()):
            if len(self.pool[subgraph]) > 1:
                overlap = Overlap(**self.initsetting)                                       # Overlap里设入initsetting
                for layerID in list(self.pool[subgraph].keys()):
                    self.pool[subgraph][layerID]['kwargs']['legend_pos'] = (layerID) * every_legend_pos # 当前图层的图例位置
                    layer = self.addLayer(self.pool[subgraph][layerID])                     # 建立图层实例
                    overlap.add(layer)
                subgraphpool[subgraph] = overlap
            else:
                for layerID in list(self.pool[subgraph].keys()):
                    self.pool[subgraph][layerID]['kwargs']['legend_pos'] = (layerID) * every_legend_pos
                    layer = self.addLayer(self.pool[subgraph][layerID])
                    subgraphpool[subgraph] = layer

        # Grid子图合成网页：如果图层数大于1，那么合成为Grid；否则直接保存
        if len(subgraphpool.keys()) > 1:
            savefile = self.addGrid(piclist=list(subgraphpool.values()), gridratio=gridratio)
        else:
            savefile = list(subgraphpool.values())[0]

        # 保存并打开
        if dirpath:
            htmlpath = safepath([dirpath, page_title+'.html'])
        else:
            htmlpath = page_title+'.html'
        savefile.render(htmlpath)
        if open_now:
            webbrowser.open(htmlpath)

# [绘图] pyecharts数据点记录类
class pyechart_markPoint():
    def __init__(self):
        self.xList = []
        self.yList = []
        self.markPointList = []

    def add(self, x, y, name=None):
        self.xList.append(x)
        self.yList.append(y)
        if not isinstance(x, str):
            x = str(x)
        self.markPointList.append({"coord": [x, y], "name": name})

    def get(self):
        return self.markPointList

# [绘图] matplotlib
def drawPicture(xaxis=None, yaxis=None, bili=None, title=[], legends=[], figurename=None):
    """
    drawPicture(xaxis=accountdata['realdate'], yaxis=(accountdata['accountequity'], movingCalmar, movingMartin))
    plt.show()
    """
    import matplotlib.pyplot as plt; import matplotlib.finance as mpf; from matplotlib.widgets import Cursor,MultiCursor; from pylab import xlim,ylim
    plt.rcParams['font.sans-serif']=['SimHei'];plt.rcParams['axes.unicode_minus']=False # matplotlib里正常显示中文标签、数字负号
    # 1.创建画布
    figurename = np.random.randint(1, 1000, 1)[0] if figurename is None else figurename
    fig = plt.figure(figurename)

    # 2.创建子图
    ax = {}; cursor = {}; n = 0
    for axnum in range(len(yaxis)):
        # 2.1 创建子图
        if bili is None:
            ax[axnum] = plt.subplot(len(yaxis), 1, axnum+1)
        else:
            ax[axnum] = plt.subplot2grid(shape=(10,1), loc=(n,0), rowspan=bili[axnum])
            n += bili[axnum]

        # 2.2 子图画折线图：如果y是tuple，则在一个子图里创建多个图; 如果y不是tuple，则直接画y
        if type(yaxis[axnum]) is tuple:
            for yn in yaxis[axnum]:
                x = range(len(yn)) if xaxis is None else xaxis
                plt.plot(x, yn, '.-')
            ylim(min(list(map(lambda x: np.nanmin(x.values) if isinstance(x,pd.core.series.Series) else np.nanmin(x),yaxis[axnum]))),
                 max(list(map(lambda x: np.nanmax(x.values) if isinstance(x,pd.core.series.Series) else np.nanmax(x),yaxis[axnum])))) # 设置y轴刻度范围
        else:
            x = range(len(yaxis[axnum])) if xaxis is None else xaxis
            plt.plot(x, yaxis[axnum], '.-')
            if isinstance(yaxis[axnum], pd.core.series.Series):
                ylim(np.nanmin((yaxis[axnum]).values), np.nanmax((yaxis[axnum]).values))
            else:
                ylim(np.nanmin(yaxis[axnum]), np.nanmax(yaxis[axnum])) # 设置y轴刻度范围

        # 2.3 2007
        plt.grid(True,color='m',linestyle=':',linewidth='0.5') # 网格
        if axnum <= len(title)-1:
            plt.title(title[axnum])  # 设置子图标题
        if axnum <= len(legends)-1:
            plt.legend([legends[axnum]], loc='best')

    # 3.
    multicursor = MultiCursor(fig.canvas, axes=[ax[n] for n in ax], color='r', lw=1, horizOn=True)    # 跨子图十字光标
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.03, top=0.97, wspace=0.06, hspace=0.06)  # 设置子图间距
    plt.show()


# [网络工具] 设置代理为shadowsocks的本地代理
def set_proxy_shadowsocks():
    """
    使用方法：要求本地打开shadowsocks（PAC或者全局模式都行），然后直接调用本函数即可。
    """
    import socket
    import socks
    socks.set_default_proxy(socks.SOCKS5, addr="127.0.0.1", port=1080)
    socket.socket = socks.socksocket



# [network] 获取本机公网IP
def get_public_network_ip():
    """获取本机公网IP"""
    from bs4 import BeautifulSoup
    from urllib.request import urlopen
    html = urlopen(r'http://ip.42.pl/raw')
    soup = BeautifulSoup(html.read(),'html5lib')
    public_network_ip = soup.text
    return public_network_ip

# [network] 设置阿里云ESC安全组配置
def set_aliyun_security_group():
    """设置阿里云ESC安全组配置

    # 依赖包：https://developer.aliyun.com/sdk?spm=5176.12818093.resource-links.dsdk_platform.488716d0Ab954C
    pip install aliyun-python-sdk-core
    pip install alibaba-cloud-python-sdk-v2
    pip install aliyun-python-sdk-core-v3
    pip install aliyun-python-sdk-ecs
    """
    from aliyunsdkcore.client import AcsClient
    from aliyunsdkcore.acs_exception.exceptions import ClientException
    from aliyunsdkcore.acs_exception.exceptions import ServerException
    from aliyunsdkecs.request.v20140526.AuthorizeSecurityGroupRequest import AuthorizeSecurityGroupRequest
    from aliyunsdkecs.request.v20140526.DescribeSecurityGroupsRequest import DescribeSecurityGroupsRequest
    from aliyunsdkecs.request.v20140526.ModifySecurityGroupRuleRequest import ModifySecurityGroupRuleRequest

    # 获取本机信息
    import socket
    host_name = socket.gethostname()                # 获取本机名

    public_network_ip = get_public_network_ip()     # 获取公网ip

    # 设置安全组
    client = AcsClient('LTAIazo9xOTgCuJt', '6esQuEd39x3y5rMICbA3HHEgGXHAtw', 'cn-hongkong')
    request = AuthorizeSecurityGroupRequest()
    request.set_accept_format('json')
    request.set_SecurityGroupId("sg-j6c6pxfkn8n418k30v3f")
    request.set_IpProtocol("tcp")
    request.set_PortRange("27017/27017")        # 要设置的端口 格式 开始端口/结束端口
    request.set_SourceCidrIp(public_network_ip) # 安全组-授权IP对象
    request.set_Policy("accept")
    request.set_Description(host_name)  # 安全组-描述信息
    request.set_Priority("1")

    response = client.do_action_with_exception(request)
    print(str(response, encoding='utf-8'))

# [network] 发送邮件
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.header import Header
class MailClient():
    """
    MailClient(sender="bobyyt@qq.com", password="vmqbocyvryycbhhh", receiver='bobyyt@qq.com').send(title='warning!', text='i am hungry', file_path='IF1809.csv')
    ps.如果想发表格，可以直接df_str = df.__str__()把表格转出str，再作为send(text=df_str)）
    """
    def __init__(self, sender, password, receiver, mailcom='QQ'):
        """
        :param sender: 发件人
        :param password: IMAP/SMTP服务授权码
        :param receiver: 收件人
        :param mailcom: 邮箱类型：QQ/163
        :return:
        """
        self.sender = sender
        self.password = password
        self.receiver = receiver
        self.mailcom = mailcom

    def send(self, title, text, file_path=None):
        message = MIMEMultipart()

        message['subject'] = title      # 标题
        message['From'] = self.sender   # 发件人 （仅作标识，无实际意义）
        message['To'] = self.receiver   # 收件人 （仅作标识，无实际意义）

        # 添加文本
        puretext = MIMEText(text)
        message.attach(puretext)

        # 添加附件
        if file_path:
            file_attch = MIMEApplication(open(file_path, "rb").read())
            file_name = os.path.split(file_path)[1]
            file_attch.add_header('Content-Disposition', 'attachment', filename=file_name)
            message.attach(file_attch)

        try:
            if self.mailcom == 'QQ':
                smtpobj = smtplib.SMTP_SSL()
                smtpobj.connect('smtp.qq.com', 465)  # 连接
            elif self.mailcom == '163':
                smtpobj = smtplib.SMTP('SMTP.163.com', 25)  # ip和端口

            smtpobj.login(self.sender, self.password)  # 登录
            smtpobj.sendmail(self.sender, self.receiver, message.as_string())  # 开始发送
            smtpobj.quit()  # 退出
            print("邮件发送成功！")
        except:
            print("失败")


# [system_time] 设置系统时间
from http import client
def setSystemTime(datetime):
    from win32api import SetSystemTime  # win32api仅仅适用于windows平台
    # tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst = time.gmtime(time.mktime(datetime.timetuple()))
    # SetSystemTime(tm_year, tm_mon, tm_wday, tm_mday, tm_hour, tm_min, tm_sec, 0)

    import os
    import time
    import ntplib
    # c = ntplib.NTPClient()
    # response = c.request('pool.ntp.org')
    # ts = response.tx_time
    ts = datetime
    _date = time.strftime('%Y-%m-%d', ts.timetuple())
    _time = time.strftime('%X', ts.timetuple())
    os.system('date {} && time {}'.format(_date,_time))

    return True

# [system_time] 恢复系统时间
def recoverSystemTime():
    from win32api import SetSystemTime  # win32api仅仅适用于windows平台
    while True:
        local_time = get_internet_time()
        if local_time:
            break
    tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst = time.gmtime(time.mktime(local_time))
    SetSystemTime(tm_year, tm_mon, tm_wday, tm_mday, tm_hour, tm_min, tm_sec, 0)
    return True

# [system_time] 从网上同步时间
def get_internet_time():
    """get beijing time by use baidu time of china"""
    conn = client.HTTPConnection("www.baidu.com")
    conn.request("GET", "/")
    r = conn.getresponse()
    # r.getheaders()            # 获取所有的http头
    ts = r.getheader('date')    # 获取http头date部分

    # 将GMT时间转换成北京时间
    ltime = time.strptime(ts[5:25], "%d %b %Y %H:%M:%S")
    ttime = time.localtime(time.mktime(ltime) + 8 * 60 * 60)
    return ttime


# [Function] 提取函数注释文档
def getFuncDoc(txt):
    """
    eg.
    txt = r'''
    '''
    getFuncDoc(txt)
    """
    # remark 只提取第二行的的def函数，要求函数前面有"   "，否则不予提取
    # remark 使用时一定要加上r'''xxxx'''转义符

    # 删除行分割线
    txt2 = txt.replace('    #----------------------------------------------------------------------\n','')

    # 添加函数文档注释
    docList = re.findall('def.*\\n.*\"\"\"(.+)\"\"\"',txt2)         # 寻找函数文档
    funcList = re.findall('(    def.*\\n.*\"\"\".+\"\"\")',txt2)    # 寻找函数

    docList2 = re.findall('class.*\\n.*\"\"\"(.+)\"\"\"',txt2)         # 寻找函数文档
    funcList2 = re.findall('(class.*\\n.*\"\"\".+\"\"\")',txt2)    # 寻找函数

    # 替换
    l = []
    txt3 = deepcopy(txt2)
    for func,doc in zip(funcList,docList):
        func2 = '    # %s\n' %(doc) + func

        if func2 in l:  # 去重
            continue
        l.append(func2)

        txt3 = txt3.replace(func, func2)

    l = []
    txt3 = txt3
    for func,doc in zip(funcList2,docList2):
        func2 = '# %s\n' %(doc) + func

        if func2 in l:  # 去重
            continue
        l.append(func2)

        txt3 = txt3.replace(func, func2)

    print(txt3)
    return txt3



"""三、优化类"""
# [并行优化] 多线程池
class WrapThread(Thread):
    # 传入semaphore
    def setSemaphore(self, sem):
        self.sem = sem

    def run(self):
        super(WrapThread, self).run()
        if self.__dict__.get('sem'):
            self.sem.release()      # 释放锁，semaphore数+1

class ThreadPool():
    def __init__(self, thread_num=500):
        """

        :param thread_num: 线程数量。注意如果thread_num=1，那么为单线程
        :return:
        """
        self.threadpool = []
        self.workerThread = None

        self.sem = multiprocessing.Semaphore(thread_num)

    def add(self, target, args=(), kwargs=None):
        """

        :param target: 目标函数
        :param args: 普通参数。type: set/list。
        :param kwargs: 关键字参数。type: dict。
        :return:
        """
        th = WrapThread(target=target, args=args, kwargs=kwargs)
        th.setSemaphore(self.sem)
        self.threadpool.append(th)

    def run(self, isJoin=True, sleep=None):
        # 新建独立线程启动线程池，防止semaphore阻塞
        self.workerThread = Thread(target=self.worker, args=(isJoin, sleep))
        self.workerThread.setDaemon(True)
        self.workerThread.start()

        if isJoin:
            self.workerThread.join()

    def worker(self, isJoin=True, sleep=None):
        for th in self.threadpool:
            self.sem.acquire()  # 获得锁，semaphore数-1
            th.setDaemon(True)
            th.start()

            if sleep:
                time.sleep(sleep)

        if isJoin:
            for th in self.threadpool:
                th.join()


# [并行优化] 多进程池
class ProcessPool():
    """
    注意事项
    * 如果是windows系统， multiprocessing.Process需在if __name__ ==  '__main__':下使用，否则会出现RuntimeError: freeze_support()
    * 在python2版本里，类的方法是不能作为target函数的

    eg.
    def testfunc(*args):
        print(args)

    tp = ThreadPool(100)
    for i in range(1000000):
        tp.add(testfunc, ('a', i))
    tp.run()
    """
    def __init__(self, process_num=None):
        self.task_list = []
        self.pool_list = []
        self.count = process_num if process_num else multiprocessing.cpu_count()

    def add(self, target, args):
        self.task_list.append([target, args])

    def run(self, isJoin=True):
        print('multiProcessPool Count: {}'.format(self.count))    # 开启的进程数

        pool = multiprocessing.Pool(self.count)
        for target, args in self.task_list:
            self.pool_list.append(pool.apply_async(target, args))
        pool.close()

        if isJoin:
            pool.join()

    def get_rusult_list(self):
        resultList = [res.get() for res in self.pool_list if res._success]   # new 修复多进程优化结果里没有成交时返回数据的问题
        resultList.sort(reverse=True, key=lambda result:result[1])
        return resultList


# [耗时分析] 耗时分析（代码段）  # remark 不推荐使用这个作为代码段耗时分析，推荐直接使用print(time.clock())
class usetime(object):
    """
    分析耗时模块。
    用法：
    a = ut.usetime()
    a.start()
    执行语句
    a.end()

    eg.
    profile = useprofile(use=True, showone=False)
    usetime = usetime()
    usetime.start()
    usetime.end()
    """
    def start(self):
        self.starttime = time.time()    #读取开始时间
        self.starttext = traceback.extract_stack()[0][1]

    def end(self):
        usetimevalue = (time.time()-self.starttime)
        self.endtext = traceback.extract_stack()[0][1]

        print('------------------------')
        print(u'执行耗时：%s秒' %usetimevalue)
        for line in range(self.starttext+1,self.endtext):
            print((linecache.getline(traceback.extract_stack()[0][0],line)).replace('\n',''))
        print('------------------------')

# [耗时分析] 耗时分析line_profiler（函数）
def useprofile(use=True, showone=True):
    """
    eg.
    @useprofile(use=True, showone=True)
    def func():
        pass
    """
    from line_profiler import LineProfiler
    LineProfilerVar = {}
    # if 'LineProfiler' in dir():
    if use is True:
        def profile(func):
            def inner(*args, **kwargs):
                if (showone is False) or (func.__name__ not in LineProfilerVar.keys()):   # 以检测过的函数就不再检测（防止多次重复检测）
                    num = func.__name__
                    LineProfilerVar[num] = LineProfiler()         # 创建实例
                    lp_wrapper = LineProfilerVar[num](func)       # 传入检测函数
                    lp_wrapper(*args, **kwargs)                   # 传入函数参数
                    LineProfilerVar[num].print_stats()            # 输出
                    print('****************************************************************************************')
                return func(*args, **kwargs)
            return inner
    else:
        def profile(func):
            def inner(*args, **kwargs):
                return func(*args, **kwargs)
            return inner
    return profile


# [缓存器] 缓存函数结果
# globals()['globals_cache_dict'] = OrderedDict()     # 全局缓存字典
def cache(key=None, keyeval=None, maxsize=10):
    """函数结果缓存器

    !notice：每个函数使用各自缓存器，缓存名的key是str((args,kwargs)) → ∴对于同一个函数来说，str((args,kwargs))相同则读取相同的值

    :param key: 缓存名（固定）
    :param keyeval: 缓存名（使用eval语法）
    :param maxsize: 缓存多少个变量
    :return:

    # 运行案例
    @cache(maxsize=10)
    def func(a, c=0, b=1):
        return a * b + c
    print func(10)          # 运行并保存到缓存 (10)
    print func(10, b=5)     # 运行并保存到缓存 (10, b=5)
    print func(10)          # 读取缓存 (10)
    print func(10, b=5)     # 读取缓存 (10, b=5)
    """
    def cache_decorator(fn):
        cache_dict = globals_cache_dict     # 获取全局缓存字典

        @wraps(fn)  # ps. wraps：让inner函数使用fn函数的doc name moude dict
        def inner(*args, **kwargs):
            """
            没有缓存：key不在缓存字典里，则执行函数并保持到字典
            有缓存：key在缓存字典里，则直接读取
            """
            if key:
                cacheName = key
            elif keyeval:
                cacheName = eval(keyeval)
            else:
                def compound_func_str(fn, args=None, kwargs=None):
                    """合成函数状态字符串  eg.func(a=1, c=3, b=2)

                    :param fn: 函数
                    :param args: 普通参数，list格式
                    :param kwargs: 关键词参数，dict格式
                    :return:

                    eg.
                    def func(a, c=0, b=1, e='a'):
                        return a * b + c
                    print compound_func_str(func, [3,1], {'b':10})
                    # 返回'func(a=3,c=1,b=10,e=a)'
                    """
                    fn_name = fn.func_name                      # 函数名
                    fn_paraList = list(fn.func_code.co_varnames)# 函数参数
                    fn_paraDict = OrderedDict()                 # 函数参数有序字典

                    # 按顺序创建参数字典
                    for k in fn_paraList:
                        fn_paraDict[k] = None

                    # 按函数默认参数填充参数字典
                    if fn.func_defaults:
                        for k,v in zip(fn_paraDict.keys()[-1*len(fn.func_defaults):], fn.func_defaults):
                            fn_paraDict[k] = v

                    # 按传入的普通参数填充参数字典
                    if args:
                        if 'self' in fn_paraList:
                            fn_paraList.remove('self')
                        for para,arg in zip(fn_paraList, args):
                            fn_paraDict[para] = arg

                    # 按关键词参数填充参数字典
                    if kwargs:
                        for para,arg in kwargs.items():
                            fn_paraDict[para] = arg

                    # 合成函数字典  eg.func(a=1, c=3, b=2)
                    l = []
                    for k,v in fn_paraDict.items():
                        para = '{}={}'.format(k,v)
                        l.append(para)
                    fn_str = '{}({})'.format(fn_name, ','.join(l))
                    return fn_str
                cacheName = compound_func_str(fn, args, kwargs)

            if cacheName not in cache_dict:
                print(u'新建缓存值: {}'.format(cacheName))
                values = fn(*args, **kwargs)
                if len(cache_dict) >= maxsize:
                    del cache_dict[cache_dict.keys()[0]]
                return cache_dict.setdefault(cacheName, values)

            else:
                print(u'获取缓存值: {}'.format(cacheName))
                return cache_dict.get(cacheName)
        return inner
    return cache_decorator


# [异常处理] 出现异常自动重新运行
def retry(fix_exception=(), fix_func=None, fix_args={}, fix_args_eval=None,
          retry_exception=(Exception), retry_times=-1, retry_delay=0):
    """
    当被main_func出现retry_exception异常重新执行该函数，fix_exception异常运行指定function

    :param fix_exception: Exception类型。使用fix_func处理后再重新执行Exception。
    :param fix_func: Function类型。出现fix_exception时，使用fix_func处理后再重新执行
    :param fix_args: dict类型。fix_func的参数
    :param fix_args_eval: str类型。当fix_func的参数需要在main_func内获取的话，传入fix_args_eval="{'普通参数args': args[0], '关键词参数kwargs': kwargs['y']}"

    :param retry_exception: Exception类型。直接重新执行的Exception。
                            ！注意：retry_exception=(Exception)时表示全部异常都重新执行。

    :param retry_times: int类型。最大重试次数。默认max_tries=-1（无限次）
    :param retry_delay: int类型。每次重试之间的延迟秒数。delay=0秒。

    :return:

    # 案例1：不对参数进行处理，仅仅执行报告的
    def fix_ZeroDivisionError_1(x, y):
        print('origin', x, y)

    # 执行函数
    @retry(fix_exception=ZeroDivisionError, fix_func=fix_ZeroDivisionError_1, fix_args_eval="{'x': args[0], 'y': kwargs['y']}",
           retry_times=5, retry_delay=3)
    def div(x, y):
        print(x / y)

    div(1, y=0)

    # -------------------------
    # 案例2：fix_func处理异常参数（有return）
    def fix_ZeroDivisionError_2(x, y):
        '''异常处理函数

        作用1：传入错误参数，修正后返回正确参数
        作用2：不返回函数，直接处理
        :param x:
        :param y:
        :return: args, kwargs 或 不返回
        '''
        # 处理结果
        print('origin', x, y)

        if y == 0:
            y += 1

        print('fix', x, y)

        # 返回正确参数
        args = ()
        kwargs = {'x': x, 'y': y}
        return args, kwargs

    # 执行函数
    @retry(fix_exception=(ZeroDivisionError), fix_func=fix_ZeroDivisionError_2, fix_args_eval="{'x': args[0], 'y': kwargs['y']}")
    def div(x, y):
        print(x / y)

    div(1, y=0)
    """

    # todo 转换为可按传入Exception顺序调用的模式
    # (({'exception': MemoryError, 'function': emailEngine.sendMail, 'args':(3, 5), 'kwargs': None, 'eval': None},
    #   {'exception': Exception, 'function': retry, 'args':(-1, ), 'kwargs': {'delay': 0},  'eval': None}))

    def wrapper(main_func):
        @wraps(main_func)
        def new_fun(*args, **kwargs):
            max_tries = retry_times     # remark while循环必须在闭包里本地引用
            while max_tries:
                # 1.正常执行就跳出循环，完成本次调用
                try:
                    return main_func(*args, **kwargs)

                # 2.1.出现fix_exception就先执行fix_func，然后继续循环，重新执行main_func
                except fix_exception as e:
                    # print出错误类型
                    print('-'*30)
                    print(e)
                    print(type(e))
                    print('-'*30)

                    # 使用eval(fix_args_eval)传入main_func的参数，然后执行fix_func
                    if fix_args_eval:
                        fix_args.update(eval(fix_args_eval))
                    fix_func_return = fix_func(**fix_args)

                    # 如果fix_func有返回值（args, kwargs)，则使用处理后的参数替换原有参数
                    if fix_func_return:
                        args, kwargs = fix_func_return

                    # 最大重试次数限制
                    max_tries -= 1
                    if not max_tries:
                        raise
                    # 延迟delay秒
                    time.sleep(retry_delay)

                    # 用continue则重新执行main_func；用break则跳出循环
                    continue    # continue 或 break

                # 2.2.出现retry_exception继续循环，重新执行main_func
                except retry_exception as e:
                    # print出错误类型
                    print('-'*30)
                    print(e)
                    print(type(e))
                    print('-'*30)

                    # 最大重试次数限制
                    max_tries -= 1
                    if not max_tries:
                        raise
                    # 延迟delay秒
                    time.sleep(retry_delay)

                    continue
        return new_fun

    return wrapper



"""四、交易相关类"""
# [KPI] 回测报告
class KPI(object):
    # notice：
    # equity、datetime传入全部要用np.array/list, 最好用pd.Series
    # 复利算法指"年复利"，按"每年进行一次复利"为标准复利基准。

    # 回报率
    def returnPercent(self, equity):
        equity = list(equity)
        rp = (equity[-1]/float(equity[0]))-1
        if rp < -1:     # FIX 最多只能亏损-100%，超过-100%返回-100%
            return -1
        return rp

    # 自然年份
    def year(self, datetime):
        datetime = list(datetime)
        return (datetime[-1] - datetime[0]).days / 365.0

    # -------------------------
    # 单利年化收益率
    def returnPercent_simpleYear(self, equity, datetime):
        return self.returnPercent(equity) / self.year(datetime)

    # 复利年化收益率
    def returnPercent_compoundYear(self, equity, datetime):
        return (1+self.returnPercent(equity))**(1/self.year(datetime)) - 1

    # -------------------------
    # 风险指标
    # 回撤率
    def drawdownPercent_list(self, equity):
        equity = pd.Series(equity)
        return (equity - equity.cummax()).div(equity.cummax()).fillna(0)  # (equity - max(equity)) / max(equity)

    # 最大回撤率
    def maxDrawdownPercent(self, equity):
        return min(self.drawdownPercent_list(equity))

    # 最大回撤率发生时间
    def maxDrawdownPercent_happenTime(self, equity, datetime):
        return datetime[np.argmin(self.drawdownPercent_list(equity))]

    # 距离上次最高点的距离
    def barslastHighestEqutiy(self, equity):
        cummaxindex = np.where(self.drawdownPercent_list(equity)==0)[0]  # 新高点的index
        barslast_highest = []
        for einx in range(len(equity)):
            if cummaxindex[cummaxindex<einx]!=[]:
                barslast_highest.append(einx - cummaxindex[cummaxindex<einx][-1]) # 上一个新高点距今的距离
            else:
                barslast_highest.append(0)
        return barslast_highest

    # 最长回撤期
    def longestDrawdownTiming(self, equity):
        return max(self.barslastHighestEqutiy(equity))

    # 最长回撤期发生时间
    def longestDrawdownTiming_happenTime(self, equity, datetime):
        return datetime[np.argmax(self.barslastHighestEqutiy(equity))]

    # -------------------------
    # 收益风险比
    # 卡玛比率calmar：(单利/复利)年化收益率/最大回撤率
    def calmar_simpleYear(self, equity, datetime):
        try:
            return self.returnPercent_simpleYear(equity, datetime) / abs(self.maxDrawdownPercent(equity))
        except ZeroDivisionError:
            print(u'最大回撤率为0，calmar_simpleYear无法计算')
            return 0

    def calmar_compoundYear(self, equity, datetime):
        try:
            return self.returnPercent_compoundYear(equity, datetime) / abs(self.maxDrawdownPercent(equity))
        except ZeroDivisionError:
            print(u'最大回撤率为0，calmar_compoundYear无法计算')
            return 0

    # 夏普比率sharpe：(单利/复利)年化收益率/年化波动率
    # interest: 年化无风险利率
    # n: 小周期转年化的可交易周期数。日转年：252或365；周转年：52；月转年：12。
    def sharpeRatio_simpleYear(self, equity, datetime, interest=0, n=252):
        barreturn = equity/equity.shift(1) - 1
        return (self.returnPercent_simpleYear(equity, datetime)-interest) / (np.std(barreturn,ddof=1) * math.sqrt(n))

    def sharpeRatio_compoundYear(self, equity, datetime, interest=0, n=252):
        barreturn = equity/equity.shift(1) - 1
        return (self.returnPercent_compoundYear(equity, datetime)-interest) / (np.std(barreturn,ddof=1) * math.sqrt(n))

    # 索提诺比率sortino：(单利/复利)年化收益率/年化下行波动率
    def downsideDev(self, equity):
        target=0
        barreturn = equity/equity.shift(1) - 1
        downsideReturn = map2(lambda x: min(target,x), barreturn)                                           # 低于target值的为下行样本
        downsideDev = math.sqrt(sum(map(lambda x: (x-target)**2, downsideReturn)) / len(downsideReturn))    # sqrt(下行样本与基准值的差的平方和 / 总样本数)
        return downsideDev

    def sortinoRatio_simpleYear(self, equity, datetime, interest=0, n=252):
        try:
            return (self.returnPercent_simpleYear(equity,datetime)-interest) / (self.downsideDev(equity)*math.sqrt(n))
        except ZeroDivisionError:
            print(u'最大回撤率为0，sortinoRatio_simpleYear无法计算')
            return 0

    def sortinoRatio_compoundYear(self, equity, datetime, interest=0,n=252):
        try:
            return (self.returnPercent_compoundYear(equity,datetime)-interest) / (self.downsideDev(equity)*math.sqrt(n))
        except ZeroDivisionError:
            print(u'最大回撤率为0，sortinoRatio_compoundYear无法计算')
            return 0

    # -------------------------
    # 综合报告
    def backtestReport(self, equity, datetime, interest=0, annualDays=252):
        # 收益指标
        return_percent = self.returnPercent(equity)
        yearreturn_simple = self.returnPercent_simpleYear(equity, datetime)
        yearreturn_compound = self.returnPercent_compoundYear(equity, datetime)

        # 风险指标
        max_drawdown_percent = self.maxDrawdownPercent(equity)
        max_drawdown_percent_time = self.maxDrawdownPercent_happenTime(equity, datetime)

        max_drawdown_bar = self.longestDrawdownTiming(equity)
        max_drawdown_bar_time = self.longestDrawdownTiming_happenTime(equity,datetime)

        #风险收益比指标
        calmar_simple = self.calmar_simpleYear(equity, datetime)
        calmar_compound = self.calmar_compoundYear(equity, datetime)

        sharpe_simple = self.sharpeRatio_simpleYear(equity, datetime, interest=interest, n=annualDays)
        sharpe_compound = self.sharpeRatio_compoundYear(equity, datetime, interest=interest, n=annualDays)

        sortino_simple = self.sortinoRatio_simpleYear(equity, datetime, interest=interest, n=annualDays)
        sortino_compound = self.sortinoRatio_compoundYear(equity, datetime, interest=interest, n=annualDays)

        #输出报告
        print('数据时段:%s - %s' %(datetime.values[0],datetime.values[-1]))
        print('总收益率:%.2f%%' %(return_percent*100))
        print('年收益率(单利):%.2f%%' %(yearreturn_simple*100))
        print('年收益率(复利):%.2f%%' %(yearreturn_compound*100))
        print('最大回撤率:%.2f%%' %(max_drawdown_percent*100))
        print(u'最大回撤率发生时间:%s' %(max_drawdown_percent_time))
        print(u'最长回撤期:%s' %(max_drawdown_bar))
        print(u'最长回撤期发生时间:%s' %(max_drawdown_bar_time))
        print('')
        print('CalmarRatio(单利):%.2f' %(calmar_simple))
        print('CalmarRatio(复利):%.2f' %(calmar_compound))
        print('SharpeRatio(单利):%.2f' %(sharpe_simple))
        print('SharpeRatio(复利):%.2f' %(sharpe_compound))
        print('SortinoRatio(单利):%.2f' %(sortino_simple))
        print('SortinoRatio(复利):%.2f' %(sortino_compound))
        print('')

        kpi = {
            'return_percent': return_percent,
            'yearreturn_simple': yearreturn_simple,
            'yearreturn_compound': yearreturn_compound,

            'max_drawdown_percent': max_drawdown_percent,
            'max_drawdown_percent_time': max_drawdown_percent_time,
            'max_drawdown_bar': max_drawdown_bar,
            'max_drawdown_bar_time': max_drawdown_bar_time,

            'calmar_simple': calmar_simple,
            'calmar_compound': calmar_compound,
            'sharpe_simple': sharpe_simple,
            'sharpe_compound': sharpe_compound,
            'sortino_simple': sortino_simple,
            'sortino_compound': sortino_compound
        }

        return kpi

    # -------------------------
    # 滚动卡玛比率calmar(单利)
    def rollingCalmar(self, equity, datetime, window):
        rollingCalmar_list = [np.NaN] * window
        for i in range(window,len(equity)):
            rollingEquity = equity[i-window:i]
            rollingDatetime = datetime[i-window:i]
            rollingCalmar = self.calmar_simpleYear(rollingEquity, rollingDatetime)
            rollingCalmar_list.append(rollingCalmar)
        return rollingCalmar_list

    # 滚动夏普比率sharpe(单利)
    def rollingSharpe(self, equity, datetime, window):
        rollingSharpe_list = [np.NaN] * window
        for i in range(window,len(equity)):
            rollingEquity = equity[i-window:i]
            rollingDatetime = datetime[i-window:i]
            rollingSharpe = self.sharpeRatio_simpleYear(rollingEquity, rollingDatetime)
            rollingSharpe_list.append(rollingSharpe)
        return rollingSharpe_list

    # 滚动索提诺比率sortino(单利)
    def rollingSortino(self, equity, datetime, window):
        rollingSortino_list = [np.NaN] * window
        for i in range(window,len(equity)):
            rollingEquity = equity[i-window:i]
            rollingDatetime = datetime[i-window:i]
            rollingSortino = self.sortinoRatio_simpleYear(rollingEquity, rollingDatetime)
            rollingSortino_list.append(rollingSortino)
        return rollingSortino_list

# [KPI] 伪回测KPI
def fake_backtest_kpi():
       import pymongo
       import pandas as pd
       import numpy as np
       import datetime as dt
       import random

       code_list = ['600000', '600004', '600009', '600010', '600011', '600015',
              '600016', '600018', '600019', '600023', '600025', '600027',
              '600028', '600029', '600030', '600031', '600036', '600038',
              '600048', '600050', '600061', '600066', '600068', '600085',
              '600089', '600100', '600104', '600109', '600111', '600115',
              '600118', '600153', '600170', '600176', '600177', '600188',
              '600196', '600208', '600219', '600221', '600233', '600271',
              '600276', '600297', '600299', '600309', '600332', '600339',
              '600340', '600346', '600352', '600362', '600369', '600372',
              '600383', '600390', '600398', '600406', '600415', '600436',
              '600438', '600482', '600487', '600489', '600498', '600516',
              '600519', '600522', '600535', '600547', '600566', '600570',
              '600583', '600585', '600588', '600606', '600637', '600660',
              '600663', '600674', '600688', '600690', '600703', '600704',
              '600705', '600733', '600741', '600760', '600795', '600809',
              '600816', '600837', '600867', '600886', '600887', '600893',
              '600900', '600919', '600926', '600958', '600977', '600998',
              '600999', '601006', '601009', '601012', '601018', '601021',
              '601066', '601088', '601108', '601111', '601117', '601138',
              '601155', '601162', '601166', '601169', '601186', '601198',
              '601211', '601212', '601216', '601225', '601228', '601229',
              '601238', '601288', '601298', '601318', '601319', '601328',
              '601336', '601360', '601377', '601390', '601398', '601555',
              '601577', '601600', '601601', '601607', '601618', '601628',
              '601633', '601668', '601669', '601688', '601727', '601766',
              '601788', '601800', '601808', '601818', '601828', '601838',
              '601857', '601877', '601878', '601881', '601888', '601898',
              '601899', '601901', '601919', '601933', '601939', '601985',
              '601988', '601989', '601992', '601997', '601998', '603019',
              '603156', '603160', '603259', '603260', '603288', '603799',
              '603833', '603858', '603986', '603993', '000001', '000002',
              '000063', '000069', '000100', '000157', '000166', '000333',
              '000338', '000402', '000408', '000413', '000415', '000423',
              '000425', '000538', '000553', '000568', '000596', '000625',
              '000627', '000629', '000630', '000651', '000656', '000661',
              '000671', '000703', '000709', '000725', '000728', '000768',
              '000776', '000783', '000786', '000858', '000876', '000895',
              '000898', '000938', '000961', '000963', '001979', '002001',
              '002007', '002008', '002010', '002024', '002027', '002032',
              '002044', '002050', '002065', '002081', '002120', '002142',
              '002146', '002153', '002179', '002202', '002230', '002236',
              '002241', '002252', '002271', '002294', '002304', '002310',
              '002311', '002352', '002410', '002411', '002415', '002422',
              '002456', '002460', '002466', '002468', '002475', '002493',
              '002508', '002555', '002558', '002594', '002601', '002602',
              '002624', '002625', '002673', '002714', '002736', '002739',
              '002773', '002925', '002938', '002939', '002945', '300003',
              '300015', '300017', '300024', '300033', '300059', '300070',
              '300072', '300122', '300124', '300136', '300142', '300144',
              '300251', '300296', '300408', '300413', '300433', '300498']

       df2 = pd.DataFrame(index=range(300))
       df2['StrategyName'] = 'strategy_dk'
       df2['backTestingTime'] = dt.datetime.now()
       df2['cumulative_returns'] = [random.uniform(-0.5, 0.7) for i in df2.index]
       df2['initCapital'] = 100000
       df2['maxDrawdown'] = [abs(min(-1*random.uniform(0.08, 0.5), i)) for i in df2['cumulative_returns'].values]
       df2['startDate'] = '2018-01-01'
       df2['endDate'] = '2019-10-01'
       df2['returnPercent_simpleYear'] = df2['cumulative_returns'] / 1.83
       df2['winningRate'] = [random.uniform(0.4, 0.8) for i in df2.index]
       df2['totalResult'] = [random.randint(3, 32) for i in df2.index]
       df2['symbol'] = code_list   # 沪深300股票代码
       df2['symbol'] = df2['symbol'].map(str)   # 沪深300股票代码

       cl = pymongo.MongoClient('47.56.102.183')
       db = cl['Vn_BackTesting']
       ct = db['stock_kpi']
       ct.insert(df2.to_dict('records'))
       df2.to_csv('stock.csv')


# [stats] 协整分析
from statsmodels.tsa.stattools import adfuller, coint
from scipy.stats import pearsonr
class cointegration():
    """
    1.读取数据
    2.清洗数据：按时间对齐 → 构建市值序列（不能直接用收盘价序列） → 对数变换（非必需）
    3.计算相关性（非必需） → 判断XY是否同阶单整 → 协整分析
    """
    def __init__(self, activeLeg_marketValue, passiveLeg_marketValue, log_transformation=False):
        """

        :param activeLeg_marketValue: 主动腿市值序列
        :param passiveLeg_marketValue: 被动腿市值序列
        :param log_transformation: 是否需要对数转换
        :return:
        """
        # 市值序列
        if not log_transformation:
            self.activeLeg_marketValue = activeLeg_marketValue
            self.passiveLeg_marketValue = passiveLeg_marketValue

        # 对数市值序列
        else:
            self.activeLeg_marketValue = activeLeg_marketValue.applymap(math.log)
            self.passiveLeg_marketValue = passiveLeg_marketValue.applymap(math.log)

    def conit(self, corr_filter=None, integrated_filter=False):
        """

        :param corr_filter: 相关性过滤参数。默认None，表示不使用相关性过滤。
        :param integrated_filter: 是否使用同阶单整判断。
        :return: isCoint, coint_pvalue
        """
        # 相关性过滤
        if corr_filter:
            corr_degrees, corr_pvalue = pearsonr(self.activeLeg_marketValue, self.passiveLeg_marketValue)
            print(u'皮尔森相关度:{}, 相关性P值: {}'.format(corr_degrees, corr_pvalue))
            if corr_pvalue < 0.05 and corr_degrees < corr_filter:
                return False, np.nan

        # 判断是否非平稳且同阶单整
        if integrated_filter:
            active_io = self.integrated_order(self.activeLeg_marketValue)
            passive_io = self.integrated_order(self.passiveLeg_marketValue)
            if not (active_io == passive_io != 0):
                print(u'不符合非平稳且同阶单整条件。active_io: {}, passive_io: {}'.format(active_io, passive_io))
                return False, np.nan

        # 协整判断
        coint_pvalue = coint(self.activeLeg_marketValue, self.passiveLeg_marketValue)[1]
        print(u'协整p-value: {}'.format(coint_pvalue))
        if coint_pvalue >= 0.05:
            return False, coint_pvalue
        else:
            return True, coint_pvalue

    # 返回单整阶数I(d)
    def integrated_order(self, data):
        d = 0
        while adfuller(data.dropna())[1] >= 0.05:
            d += 1
            data = data.diff()
        return d

    # 回归分析OLS
    def ols(self):
        constant, ols = (sm.OLS(self.passiveLeg_marketValue, sm.add_constant(self.activeLeg_marketValue))).fit().params
        return ols, constant


# [okex] okex交割合约代码计算（当周、次周、当季）
def checkDelivery_OKEX(run_datetime):
    """
    :param run_datetime: 运行时间，dt.datetime格式
    :return:
    """
    # 变量
    startdate = run_datetime.date()
    enddate = startdate + dt.timedelta(weeks=18)
    date_range = pd.date_range(startdate, enddate)

    # 周五列表
    friday_range = filter(lambda date: date if dt.datetime.weekday(date)==4 else None, date_range)  # [Timestamp('2019-03-29 00:00:00', freq='D'), Timestamp('2019-04-05 00:00:00', freq='D'), ...]

    # 季度列表
    quarter_dict = {}
    for date in friday_range:
        if date.month in [3,6,9,12]:
            quarter_dict[date.month] = date
    quarter_range = quarter_dict.values()   # Timestamp('2019-06-28 00:00:00', freq='D')], [Timestamp('2019-03-29 00:00:00', freq='D'),
    quarter_range.sort()    # [Timestamp('2019-03-29 00:00:00', freq='D'), Timestamp('2019-06-28 00:00:00', freq='D')]

    # 周五16:00前取当周，否则取下周时间
    if run_datetime < dt.datetime.combine(friday_range[0], dt.time(16)):
        thisweek = friday_range[0]
        nextweek = friday_range[1]
    else:
        thisweek = friday_range[1]
        nextweek = friday_range[2]

    if nextweek < quarter_range[0]:
        quarter = quarter_range[0]
    else:
        quarter = quarter_range[1]

    thisweek = thisweek.strftime('%y%m%d')
    nextweek = nextweek.strftime('%y%m%d')
    quarter = quarter.strftime('%y%m%d')

    print(u'OKEXF: %s, 当周%s, 次周%s, 当季%s' %(dt.datetime.strftime(run_datetime, '%Y%m%d %H:%M:%S'), thisweek, nextweek, quarter))
    return thisweek, nextweek, quarter


# [vnpy2] 使用优化参数的csv文件设置cta_setting.json
def set_cta_setting_json_from_optimize_csv_vnpy2(optimize_csv_path, strategy_name):
    """使用优化参数的csv文件设置cta_setting.json

    :param optimize_csv_path: 优化参数的csv文件夹路径
    :param strategy_name: 策略名
    :return:

    eg.
    set_cta_setting_json_from_optimize_csv(optimize_csv_path=r'G:\test', strategy_name='QianKunStockStrategy')
    """
    strategy_dict = {}

    l = getDirFile(optimize_csv_path)
    path_list = [i[2] for i in l]
    for path in path_list:
        df = pd.read_csv(path)
        gb = df.groupby('symbol').first()
        print(df)

        for i,d in gb.iterrows():
            instance_name = '{}_{}'.format(strategy_name, i)
            class_name = instance_name
            vt_symbol = i

            param_list = list(d.keys())
            param_list.remove('target')
            param_list.remove('zscore')

            setting={}
            setting['class_name'] = class_name
            for param_name in param_list:
                setting[param_name] = d[param_name]

            strategy_dict[instance_name] = dict(class_name=strategy_name, vt_symbol=vt_symbol, setting=setting)

    with open('cta_setting.json', 'w+') as f:
        j = json.dumps(strategy_dict, indent=4)
        f.write(j)

# [vnpy2] 使用优化参数的csv文件设置backtesting_dict
def set_backtesting_dict_from_optimize_csv_vnpy2(optimize_csv_path, strategy_name):
    """使用优化参数的csv文件设置backtesting_dict

    :param optimize_csv_path: 优化参数的csv文件夹路径
    :param strategy_name: 策略名
    :return:

    eg.
    set_backtesting_dict_from_optimize_csv_vnpy2(optimize_csv_path=r'G:\test', strategy_name='QianKunStockStrategy')
    """
    strategy_dict = {}

    l = getDirFile(optimize_csv_path)
    path_list = [i[2] for i in l]
    for path in path_list:
        df = pd.read_csv(path)
        gb = df.groupby('symbol').first()

        for i,d in gb.iterrows():
            vt_symbol = i

            param_list = list(d.keys())
            param_list.remove('target')
            param_list.remove('zscore')

            setting={}
            for param_name in param_list:
                setting[param_name] = d[param_name]
            strategy_dict[vt_symbol] = setting
    print(strategy_dict)

# [vnpy] bar类模仿tick类 # remark 单个数据类
def simulate_1minBar_tick(bar):
    """把VtBarData()模拟为VtTickData()

    :param bar:
    :return:
    """
    tick = deepcopy(bar)
    tick.lastPrice = bar.close
    tick.volume = bar.volume
    tick.datetime = bar.datetime - dt.timedelta(seconds=60)    # tick.datetime是按达到时间算，bar.datetime是按bar的开始时间算，∴要减去1分钟的时间
    return tick

# [vnpy] bar类模仿tick类 # remark pd.DataFrame版
def simulate_1minBarData_tickData(bardata):
    """把VtBarData的DataFrame 模拟为 VtTickData()的DataFrame

    :param bardata:
    :return:

    eg.
    cl = pymongo.MongoClient('47.56.102.183')
    db = cl['VnTrader_1Min_Db']
    ct = db['BTC-USD-191018.OKEX']
    cursor = ct.find().sort('datetime')
    bar_data = pd.DataFrame(list(cursor))
    tick_data = simulate_1minBarData_tickData(bar_data)
    print(tick_data)
    """
    # 按VtTickData的属性和默认值构建tickdata
    from vnpy.trader.vtObject import VtTickData
    tickdata = pd.DataFrame([VtTickData().__dict__]*len(bardata))

    # 把相同列传入值
    bar_columns = bardata.columns
    tick_columns = tickdata.columns
    for c in bar_columns:
        if c in tick_columns:
            tickdata[c] = bardata[c]

    # 把bar的属性模拟为tick的属性
    tickdata['lastPrice'] = bardata['close']
    tickdata['datetime'] = bardata['datetime'] + dt.timedelta(seconds=60)  # tick.datetime是按达到时间算，bar.datetime是按bar的开始时间算，∴要在bar.datetime+1分钟的时间

    return tickdata

# [vnpy] 判断主力合约 (from:汪振宇)
class Active(object):
    # todo 未完成
    # 实盘运行时获取当前的主力合约（放在DrEngine.loadSetting()里）
    # 1.连接mongodb['VnTrader_Daily_Db']，获取当前品种的所有合约名（品种'IF'，获得所有合约名[IF18xx~IF19xx, IF.HOT]）
    # 2.读取所有合约的上一个交易日的数据
    # 3.
    def __init__(self, path):
        from collections import OrderedDict, defaultdict
        from vnpy.trader.gateway.ctpGateway.ctpGateway import exchangeMapReverse
        import re

        print(u"开始打开文件")
        with open(path) as f:
            self.file = f.read()
        self.prices = list()
        self.change = list()
        self.active = list()
        self.active_group = defaultdict(list)
        self.judge_exchange()
        self.judge_active()

    # 提取各交易所合约信息
    def judge_exchange(self):
        msg = json.loads(self.file)
        # 使用正则分类同一品种合约
        for exchange in exchangeMapReverse.values():
            if exchange in msg:
                # 得到合约信息列表
                value = msg[exchange]
                # 添加的是字典 "IC1811": [22269.0,"20181116"]，。。。
                self.change.append(value)

    # 正则匹配合约品种
    def judge_active(self):
        for contracts in self.change:
            for contract, openInterest_and_time in contracts.items():
                if contract == "scefp":
                    continue
                # 正则匹配合约代码
                symbol = re.findall(r"(.*?)\d", contract)[0]
                # 分组字典
                #  # "IC": [ IC1811:[22269.0,"20181116"，。。。]
                self.active_group[symbol].append({contract: openInterest_and_time})
        self.write_contract()

    # todo 主力判断方式
    # 现况：先按时间排序，再按成交量排序。
    # 目标：1.取同品种的"合约交割日>当前主力交割日"的合约 2.取其上一交易日的dailyVol,dailyOI 2.以满足"dailyVol>主力dailyVol, dailyOi>主力dailyOi*1.1"的、"交割日最近"的合约为主力合约

    # 判断主力合约，并且写入文件
    def write_contract(self):
        #  # "IC": [ IC1811:[ 22269.0,"20181116"]。。。]
        for variety, items in self.active_group.items():
            # variety 品种名， items 每个品种下的合约和持仓以及日期数据
            # items =         [ IC1811:[ 22269.0,"20181116"...,]
            for datas in items:
                for contract, data in datas.items():
                    # 下方添加后的格式 [ (IC1811，[22269.0,"20181116"] )....]
                    self.prices.append((contract, data))
            # 拿到最大的持仓合约 # 先按合约的持仓排序
            self.prices.sort(key=lambda x: x[1][0], reverse=True)
            # 只取该品种第一个
            self.active.append({variety + ".HOT": self.prices[0][0]})
        # 先按成就量排序
        # active = {"active": {y.items()[0][0]: y.items()[0][1] for y in self.active}}
        # active["active"] = dict(active["active"])
        with open("DR_setting.json", "r+") as f:
            # {"active": {"IC.HOT": "IC1712",IF.HOT": "IF1712",...}}
            drSetting = json.load(f)
            # 比对日期在决定是否更换主力合约
            # drSetting["active"] = active["active"]
            old_active = drSetting["active"]
            for contract, symbol in old_active.items():
                for new_symbol in self.prices:
                    # 正则判断合约代码在是否更新
                    if re.findall(r"(.*)?\.", contract)[0] == re.findall(
                            r"(.*)?\d", new_symbol[0])[0] and symbol != \
                            new_symbol[0]:
                        old_active[contract] = new_symbol[0]
            drSetting["active"] = old_active
            data = json.dumps(drSetting, sort_keys=True, indent=4)
            f.seek(0, 0)
            f.write(data)

# [vnpy] 传入vnpy的vtSymbol，判断属于哪个交易所
def getVtSymbolExchange(VtSymbol):
    x = VtSymbol.split('.')
    if len(x)>=2 and x[1]!='HOT':
        return x[1]
    elif len(x)==1 or x[1]=='HOT':
        return 'CTP'


# [QuantDog] QuantDog的实盘策略：胜率统计
def QD_winningRate():
    cl = pymongo.MongoClient('47.75.181.114')
    ct = cl['vn_okexf']['vn_okexf_strategy_local_orders']
    symbolList = ['BTC', 'BCH', 'LTC', 'ETH', 'ETC', 'EOS']
    for symbol in symbolList:
        flt = {'productClass': 'TripleMaStrategy', 'currency': symbol, 'offset': u'平仓', 'orderDateTime': {'$gte': '20190807'}}
        l = list(ct.find(flt))
        if not l:
            print('not {}'.format(symbol))
            continue

        df = pd.DataFrame(l)
        df['dir'] = map(lambda x: 1 if x == u'空' else -1, df['direction'])
        df['pnl'] = (df['price'] - df['entryPrice']) * df['dir']

        # # 只统计最近30条交易记录的数据
        # lastTradeNum = 30
        # df = df[-lastTradeNum:]

        win_ = len(df.ix[df['pnl'] > 0])
        all_ = len(df['pnl'])
        wRatio = win_ / all_
        # print(df)
        print(symbol, wRatio, win_, all_)

# [datetime] 用datetime合成tradedate
def turnDatetimeToTradedate_CTP(datetime, is_datetime_only_date=False):
    """
    :param datetime: 格式为date+time的时间列 eg. 2017-11-30 20:59:00
    :param is_datetime_only_date: datetime是不是只有date而无time。常见于日线数据。  eg.2017-11-30
    :return: 返回tradedate的list
    """
    # 1.如果datetime只有date无time（一般是日/周/月线数据），那么直接取当前date
    if is_datetime_only_date:
        tradedate = map2(lambda x: x.date(), datetime)
    else:
        # 2.如果datetime在8:00-16:00（早盘9:00提前一小时、收盘15:00晚一小时），那么直接取datetime.date，否则取0
        tradedate = map2(lambda x: x.date() if ((x.time()>=dt.time(8,00)) and (x.time()<dt.time(16,00))) else np.nan, datetime)
        # 3.datetime不在8:00-16:00（tradedate为0的部分），则取下一个9:00-21:00的date（不能直接取实际值，因为ag等夜盘时间有到周六的2:00才结束的）
        data = pd.DataFrame()
        data['datetime'] = datetime
        data['tradedate'] = tradedate
        data = data.fillna(method='bfill')
        tradedate = data['tradedate'].tolist()
    return tradedate

