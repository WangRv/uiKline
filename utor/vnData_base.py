# coding: utf-8
from __future__ import print_function, division
from utFunction import *

import bson
import pymongo
import datetime as dt
import time
import requests
import logging
from pymongo.errors import DuplicateKeyError, AutoReconnect, NetworkTimeout
from vnpy.trader.vtObject import *


# [索引] 创建索引
def creat_index(collection, index_key, sortDirection=1):
    indexs_now = collection.index_information().keys()

    # 如果index_key不在collection，那么
    if '{}_{}'.format(index_key, sortDirection) in indexs_now:
        collection.create_index([(index_key, sortDirection)])

    return collection

# [索引] 删除索引
def drop_index(collection, index_key):
    indexs_now = collection.index_information().keys()

    if '{}_1'.format(index_key) in indexs_now:
        collection.drop_index('{}_1'.format(index_key))

    elif '{}_-1'.format(index_key) in indexs_now:
        collection.drop_index('{}_-1'.format(index_key))


# [读取] 聚合读取数据（推荐）
def mongo_aggregate(collection, match=None, project=None, sort=None, group=None):
    """

    :param collection: pymongo.collection。

    :param match: dict。
    eg.{'datetime': {'$gte': dt.datetime(2019, 9, 20, 8, 0)}}

    :param project: dict。
    eg.{"$project": {'_id': 0}}

    :param sort: dict。
    eg.{'datetime': 1}

    :param group: dict。
    eg.{'_id': 'vtSymbol', 'close_avg': {'$avg': 'close'}}

    :return: cursor
    """
    pipeline = []

    if match:
        pipeline.append({"$match": match})

    if project:
        pipeline.append({"$project": project})

    if sort:
        pipeline.append({"$sort": sort})

    if group:
        pipeline.append({"$group": group})

    cursor = collection.aggregate(pipeline, allowDiskUse=True)
    return cursor

# [读取] 分页读取数据   # remark 不推荐。sort时还是容易超出内存限制
def mongo_batch(collection, flt=None, sort=None, size=1000):
    """

    :param collection: pymongo.collection
    :param flt: collection取出的数据范围
    :param sort: 排序。格式：[('datetime', 1)]
    :param size: 拼合的data的size
    :return:

    # 使用案例：
    ct = pymongo.MongoClient('47.56.102.183')['VnTrader_Tick_Db']['BTC-USD-191018.OKEX']
    batch = mongo_batch(ct, flt={'datetime': {'$gte': dt.datetime(2019,10,17,14), '$lte': dt.datetime(2019,10,17,16)}}, sort=[('datetime', 1)], size=100)
    for data in batch:
        print(data)
    """
    if not flt:
        flt = {}

    if not sort:
        sort = {}

    cursor = collection.find_raw_batches(flt, sort=sort, no_cursor_timeout=True).batch_size(size)   # remark sort操作如果没有索引，有可能超出内存限制
    for batch in cursor:
        batchdata = pd.DataFrame(bson.decode_all(batch))
        yield batchdata
    cursor.close()


# [复制] 复制database
def copy_database(client, fromdb, todb, fromhost='127.0.0.1'):
    """

    :param client: 目标mongo client。mongo client类。
    :param fromdb: 源database名。str。
    :param todb: 目标database名。str。
    :param fromhost: 源mongo host。str。
    :return:

    eg.
    cl = pymongo.MongoClient('localhost')
    copy_database(cl, 'VnTrader_1Min_Db_CLEAN', 'VnTrader_1Min_TEST')
    """
    client.admin.command('copydb', fromdb=fromdb, todb=todb, fromhost=fromhost)     # 注意：要求目标database没有与原database重复的collection

# [复制] 跨服务器复制collection
def copy_collection_diffServer(client, fromdbName, fromctName, fromhost='127.0.0.1', copyIndexes='false'):
    """

    :param client: 目标client
    :param fromdbName: 源database名。str。
    :param fromctName: 源collection名。str。
    :param fromhost: 源mongo host。str。
    :param copyIndexes: 是否复制源collection的索引。str：'false' or 'true'。
    :return:

    eg.
    cl = pymongo.MongoClient('localhost')
    copy_collection_diffServer(cl, 'VnTrader_1Min_Db', 'btcusdt.HUOBI', '47.52.254.220')
    """
    from collections import OrderedDict
    config = OrderedDict((
        ("cloneCollection", "{}.{}".format(fromdbName, fromctName)),
        ("from", "{}:27017".format(fromhost)),
        ("copyIndexes", copyIndexes)
    ))
    database = client['a']  # remark：因为只能复制到目标client的同名collecion里，所以连接随便一个database即可。
    database.command(config)


# [备份] 生成mongodump命令行
def mongodump_command(mongohost, database, collection, startdate, enddate, savepath):
    """
    eg.
    db = pymongo.MongoClient('47.56.102.183')['VnTrader_Tick_Db']
    ctNameList = db.collection_names()
    ctNameList.sort()
    ctNameList = filter(lambda x: x if getVtSymbolExchange(x) == 'OKEX' else None, ctNameList)
    for ctName in ctNameList:
        mongodump_command('47.56.102.183', 'VnTrader_Tick_Db', ctName, '20190901', '20191001', r'D:\mongo_gz')
    """
    starttime = dt.datetime.strptime(startdate, '%Y%m%d')
    endtime = dt.datetime.strptime(enddate, '%Y%m%d')

    starttime_unix = int(time.mktime(starttime.timetuple())) * 1000
    endtime_unix = int(time.mktime(endtime.timetuple())) * 1000

    txt = r'''mongodump --host %s --db %s --collection %s -q "{"datetime":{$gte:Date(%s), $lt:Date(%s)}}" --gzip -o %s''' % (mongohost, database, collection, starttime_unix, endtime_unix, savepath)
    print(txt)

# [备份] 生成mongorestore命令行
def mongorestore_command(database, dirpath=None, filepath=None):
    """
    eg.
    mongorestore_command('VnTrader_Tick_Db', r'D:\mongo_gz')
    """
    if dirpath:
        l = getDirFile(dirpath)
        l2 = [set_[2] for set_ in l if 'bson.gz' in set_[2]]

        for filepath in l2:
            txt = r'''mongorestore --gzip --db %s %s''' % (database, filepath)
            print(txt)

    elif filepath:
        txt = r'''mongorestore --gzip --db %s %s''' % (database, filepath)
        print(txt)


# [合成] tick合成1MinBar（vnpy版）
def transform_tick_1MinBar(data_tick):
    period = '1T'

    data_tick = deepcopy(data_tick)
    data_tick.index = data_tick['datetime']

    data_1T = pd.DataFrame(columns=['datetime','date','time'])

    data_1T['open'] = data_tick['lastPrice'].resample(period).first()
    data_1T['high'] = data_tick['lastPrice'].resample(period).max()
    data_1T['low'] = data_tick['lastPrice'].resample(period).min()
    data_1T['close'] = data_tick['lastPrice'].resample(period).last()
    data_1T['volume'] = data_tick['volume'].resample(period).sum()
    data_1T['openInterest'] = data_tick['openInterest'].resample(period).last()

    data_1T['vtSymbol'] = data_tick['vtSymbol'].resample(period).first()
    data_1T['symbol'] = data_tick['symbol'].resample(period).first()
    data_1T['gatewayName'] = data_tick['gatewayName'].resample(period).first()
    data_1T['exchange'] = data_tick['exchange'].resample(period).first()

    data_1T['datetime'] = data_1T.index
    data_1T['date'] = data_1T['datetime'].map(lambda x: x.strftime('%Y%m%d'))
    data_1T['time'] = data_1T['datetime'].map(lambda x: x.strftime('%H:%M:%S.%f'))

    if 'fee' in data_tick.columns:
        data_1T['fee'] = data_tick['fee'].resample(period).mean()

    data_1T = data_1T.dropna(thresh=5)
    data_1T = data_1T.reset_index(drop=True)
    return data_1T

# [合成] 合成xmin的bar
def transform_1MinBar_xMinBar(data_1T, period, isCTP=False):
    data_1T = deepcopy(data_1T)

    if isCTP and period[-1] in ['D','W','M','A']:
        data_1T['tradedate'] = turnDatetimeToTradedate_CTP(data_1T['datetime'])
        data_1T.index = data_1T['tradedate']    # CTP的日线以上周期转换，需要用tradedate来转换
    else:
        data_1T.index = data_1T['datetime']
    data_1T.index = pd.to_datetime(data_1T.index)

    data1 = pd.DataFrame(columns=['datetime','date','time'])

    data1['open'] = data_1T['open'].resample(period).first()
    data1['high'] = data_1T['high'].resample(period).max()
    data1['low'] = data_1T['low'].resample(period).min()
    data1['close'] = data_1T['close'].resample(period).last()
    data1['volume'] = data_1T['volume'].resample(period).sum()
    data1['openInterest'] = data_1T['openInterest'].resample(period).last()

    data1['vtSymbol'] = data_1T['vtSymbol'].resample(period).first()
    data1['symbol'] = data_1T['symbol'].resample(period).first()
    data1['gatewayName'] = data_1T['gatewayName'].resample(period).first()
    data1['exchange'] = data_1T['exchange'].resample(period).first()
    data1['rawData'] = data_1T['rawData'].resample(period).first()

    data1['datetime'] = data1.index
    data1['date'] = data1['datetime'].map(lambda x: x.strftime('%Y%m%d'))
    data1['time'] = data1['datetime'].map(lambda x: x.strftime('%H:%M:%S.%f'))

    data1 = data1.dropna(thresh=5)
    data1 = data1.reset_index(drop=True)
    return data1

# [合成] 把tickdata转换为renkodata     # todo 这是绝对值版本，增加百分比版本
class transform_tick_renko():
    """
    # 使用案例：
    from vnpy.utor.vnData_base import mongo_aggregate

    cl = pymongo.MongoClient('47.56.102.183')
    db = cl['VnTrader_Tick_Db']
    ct = db['BTC-USD-191018.OKEX']
    cursor = mongo_aggregate(ct, match={'datetime': {'$gte': dt.datetime(2019, 10, 17, 14), '$lte': dt.datetime(2019, 10, 17, 16)}}, sort={'datetime': 1})
    tick_data = pd.DataFrame(list(cursor))

    tr = tick_to_renko(tick_data, 10)
    origin_renko_data = tr.get_origin_renko_data()  # renko化的原data
    renko_data = tr.get_renko_data()    # 完全的renkodata
    print(origin_renko_data)
    print(renko_data)
    """
    def __init__(self, tickdata, brickSize, inversion=True, gap=False, brick_attr_list=None):
        """影响因素：是否反转，是否记录幽灵renko（一个tick穿越多个brick），

        :param tickdata: tickdata，字段为['datetime', 'lastPrice', 'openInterest', 'lastVolume', 'volume', 'bidPrice1', 'askPrice1']
        eg. tickdata = pd.DataFrame(data=[], columns=['datetime', 'lastPrice', 'openInterest', 'lastVolume', 'volume', 'bidPrice1', 'askPrice1'])
        :param brickSize: 价格每波动brickSize，就构造一个renko
        :param inversion: 是否展示反转renko。False：非反转版：只要波幅达到renkopip就生成新renko；True：反转版本：(与上一根renko)方向相反的波幅要达到2个renkopip才生成新的反向renko
        :param gap: 是否展示幽灵renko（即一个tick穿越多个brickSize时，是否展示这些不可交易的的brick）
        :param brick_attr_list: [point, pointup, pointdw]。多数据拼接专用，用前期数据的[point,pointup,pointdw]作为计算renkopoint的起始点。
        :return: data
        """
        # 参数
        self.tickdata = tickdata
        self.brickSize = brickSize
        self.inversion = inversion
        self.gap = gap
        self.brick_attr_list = brick_attr_list

        # origin_renko_data（即截取原数据的穿越tickdata，并增加部分renko属性）
        self.origin_renko_data = self.calculate_origin_renko_data(self.tickdata)

    # 1.获得point形式的renkodata
    def calculate_origin_renko_data(self, tickdata):
        tickdata = deepcopy(tickdata)

        # 1.获得临界tick（穿越&碰到point位的tick）
        tickdata['floor'] = np.floor(tickdata['lastPrice'] / self.brickSize)
        tickdata['floor_diff'] = tickdata['floor'].diff()

        cross_index1 = np.where(tickdata['floor_diff']!=0)[0]         # 提取发生floor变化的tick
        cross_index2 = np.where(tickdata['floor_diff']<0)[0]-1        # 提取floor向下穿越的tick的上一个tick 【目的：把向下碰线的tick也算进来。eg.332-330-329的floor是330-330-320，floor_diff算法只会提取329这个tick而忽略330，而330这个碰线tick会被忽略】
        cross_index = np.union1d(cross_index1, cross_index2)          # 取两者交集并排序

        cross_data = tickdata.ix[cross_index,:]
        cross_data = cross_data.reset_index()

        # 2.缓存：初始化brickBase、brick_up、brick_dw
        if self.brick_attr_list is None:
            brick_base = cross_data.loc[0, 'floor'] * self.brickSize
            brick_up = brick_base + self.brickSize
            brick_dw = brick_base - self.brickSize
            brick_index = [0]
            brick_base_list = [brick_base]
        else:
            brick_base, brick_up, brick_dw = self.brick_attr_list
            brick_index = []
            brick_base_list = []

        # 3.判断是否是穿越tick（是否穿越pointup/brick_dw，以及是否一个tick穿越了多个renko）
        for inx in cross_data.index:
            if cross_data.loc[inx,'lastPrice'] >= brick_up:          # 说明：之所以要加上判断条件，是因为如果触发了price>=pointup后pointdw就发生变化了，所以需要用if...else...把price>=pointup和price<=pointdw分流为两个分支。
                while cross_data.loc[inx,'lastPrice'] >= brick_up:   # 如果一bar穿越多个point，那么把n个point都记录下
                    brick_index.append(inx)
                    brick_base_list.append(brick_up)
                    brick_base = brick_up
                    brick_up = brick_base + self.brickSize
                    brick_dw = (brick_base - 2 * self.brickSize) if self.inversion else (brick_base - self.brickSize)  # 如果要反转，则要达到2倍brickSize才记录

            elif cross_data.loc[inx,'lastPrice'] <= brick_dw:
                while cross_data.loc[inx,'lastPrice'] <= brick_dw:
                    brick_index.append(inx)
                    brick_base_list.append(brick_dw)
                    brick_base = brick_dw
                    brick_up = (brick_base + 2 * self.brickSize) if self.inversion else (brick_base + self.brickSize)
                    brick_dw = brick_base - self.brickSize

        # 4.截取原数据的穿越tick，并增加部分属性
        crosstick_inx = cross_data.loc[brick_index,'index'].tolist()    # renko的crosstick在tickdata的index
        crosstick_data = pd.DataFrame()
        if crosstick_inx:
            crosstick_data = self.tickdata.loc[crosstick_inx, :].reset_index(drop=True) # 截取原数据结束renko时的tick
            crosstick_data['crosstick_idx'] = crosstick_inx     # 原数据的index
            crosstick_data['brick_base'] = brick_base_list      # renko的brick_base
            crosstick_data['isgap'] = np.where(crosstick_data['crosstick_idx']==crosstick_data['crosstick_idx'].shift(-1), 'yes', 'no')    # 判断是否幽灵renko

        # *多数据拼接：外输brick_base、brick_up、brick_dw
        self.brick_attr_list = [brick_base, brick_up, brick_dw]
        return crosstick_data

    # 返回origin_renko_data（即截取原数据的穿越tickdata，并增加部分renko属性）
    def get_origin_renko_data(self):
        return self.origin_renko_data

    # 返回当前数据段的brick_attr    # remark 用于多数据拼接
    def get_brick_attr(self):
        return self.brick_attr_list

    # ----------------------------
    # 2.转换为OHLC格式的RENKO
    def calculate_ohlc_renkodata(self, crosstick_data):
        """规则：以结束时点为准

        1.renko的datetime是renko的结束时间。
        2.open是上一个renko结束时的价格，close是当前renko结束时的价格。high、low取前两者的最大最小值。
        3.volume是renko之间的成交量累计。
        """
        renko_data = deepcopy(crosstick_data)

        # 删除跳空renko
        if not self.gap:
            gap_index = np.where(renko_data['isgap'] == 'yes')
            if gap_index:
                renko_data.drop(gap_index[0], axis=0, inplace=True)
                renko_data.reset_index(drop=True, inplace=True)

        # 行情
        renko_data['open'] = renko_data['lastPrice'].shift(1)
        renko_data['close'] = renko_data['lastPrice']
        # renko_data['high'] = map2(lambda x,y: max(x,y), renko_data['open'], renko_data['close'])
        # renko_data['low'] = map2(lambda x,y: min(x,y), renko_data['open'], renko_data['close'])

        tickdata_price = self.tickdata['lastPrice'].values
        idx = renko_data['crosstick_idx'].values
        idx_set = [(x,y) for x,y in zip(idx[:-1], idx[1:])]
        renko_data['high'] = [np.nan] + map(lambda s: max(tickdata_price[s[0]:s[1]]), idx_set)
        renko_data['low'] = [np.nan] + map(lambda s: min(tickdata_price[s[0]:s[1]]), idx_set)

        # 成交量   # remark ∵tick_data['volume']是当日累计成交量，在隔夜时成交量会清空，导致隔夜renko的成交量出错。因此需要先计算总表的累计成交量，再在两个renko之间的累计成交量相减
        self.tickdata['sumvol'] = self.tickdata['volume'].cumsum()
        renko_sumvol = self.tickdata.loc[renko_data['crosstick_idx'],'sumvol']
        renko_data['volume'] = (renko_sumvol - renko_sumvol.shift(1)).values

        # 特殊
        renko_data['open_ask'] = renko_data['askPrice1'].shift(1)
        renko_data['open_bid'] = renko_data['bidPrice1'].shift(1)
        renko_data['close_ask'] = renko_data['open_ask']
        renko_data['close_bid'] = renko_data['open_bid']

        # if 'tradedate' not in renko_data.columns:
        #     renko_data['tradedate'] = renko_data['datetime'].map(lambda x: x.date().strftime('%Y%m%d'))

        # 删除无用行数
        renko_data = renko_data[['datetime', 'open', 'high', 'low', 'close', 'volume', 'openInterest',
                                 'brick_base', 'open_ask', 'open_bid', 'close_ask', 'close_bid']]
        # renko_data.dropna(axis=0, how='any', inplace=True)
        # renko_data.reset_index(drop=True, inplace=True)

        return renko_data

    # 返回renkodata（完全按renko属性的构造的data）
    def get_renko_data(self):
        renkodata = self.calculate_ohlc_renkodata(self.origin_renko_data)
        return renkodata


# [其他] 获得MongoClient下的每一个dbname,ctname
def get_all_name(client):
    l = []
    dbnameList = client.database_names()
    for dbname in dbnameList:
        db = client[dbname]
        ctnameList = db.collection_names()
        ctnameList.sort()
        for ctname in ctnameList:
            l.append([dbname, ctname])
    return l

# [其他] 数据状态记录
class DataStatus():
    def __init__(self, collection_status, collection_origin, collection_target):
        self.collection_status = collection_status

        self.collection_origin_host = collection_origin.database.client.HOST
        self.collection_origin_name = collection_origin._Collection__full_name
        self.collection_target_host = collection_target.database.client.HOST
        self.collection_target_name = collection_target._Collection__full_name

    def save_status(self, start, end, status):
        """

        :param start: dt.datetime
        :param end: dt.datetime
        :param status: str
        :return:
        """
        d =   { 'collection_origin_host': self.collection_origin_host,
                'collection_origin_name': self.collection_origin_name,
                'collection_target_host': self.collection_target_host,
                'collection_target_name':self.collection_target_name,
                'start': start if isinstance(start, dt.datetime) else dt.datetime.strptime(start, '%Y%m%d %H:%M'),
                'end': end if isinstance(end, dt.datetime) else dt.datetime.strptime(end, '%Y%m%d %H:%M'),
                'status': status}
        self.collection_status.insert_one(d)

    def load_status(self, start, end, status=None):
        """

        :param start: dt.datetime
        :param end: dt.datetime
        :param status: str
        :return:
        """
        flt = { 'collection_origin_host': self.collection_origin_host,
                'collection_origin_name': self.collection_origin_name,
                'collection_target_host': self.collection_target_host,
                'collection_target_name':self.collection_target_name,
                'start': {'$gte': start if isinstance(start, dt.datetime) else dt.datetime.strptime(start, '%Y%m%d %H:%M')},
                'end': {'$lt': end if isinstance(end, dt.datetime) else dt.datetime.strptime(end, '%Y%m%d %H:%M')}}
        if status:
            flt.update({'status': status})

        data = pd.DataFrame(list(self.collection_status.find(flt).sort('start')))

        if len(data):
            date_range = zip(data['start'], data['end'])
            return date_range
        else:
            return []

# [下载] 补充缺失的历史数据（未完成）
def supplementaryData():
    # 从数据库读取数据
    initData = self.loadBar('20190405', '20190416')
    a = map(lambda x: x.datetime, initData)
    b = pd.date_range('20190405', '20190416', freq='1T')
    c = filter(lambda x: x if x not in a else None, b)
    for i in c:
        print(i)




