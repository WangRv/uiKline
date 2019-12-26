# coding: utf-8
from __future__ import print_function, division
from vnData_base import *

#================================================
"""数据处理"""
# 发送连接错误的邮件
def sendMail_connectError(text=''):
    mc = MailClient(sender="bobyyt@qq.com", password="vmqbocyvryycbhhh", receiver='bobyyt@qq.com')
    mc.send(title=u'vnDataClean运行出错', text=text)

# vnpy数据处理
class VnTickData_Clean():
    """案例
    l = ['BTC-USD-SWAP.OKEX', 'BCH-USD-SWAP.OKEX', 'LTC-USD-SWAP.OKEX', 'ETH-USD-SWAP.OKEX', 'ETC-USD-SWAP.OKEX', 'EOS-USD-SWAP.OKEX', 'XRP-USD-SWAP.OKEX']
    date_range = pd.date_range('20190601', '20190604', freq='5T')
    for symbol in l:
        collection_origin = pymongo.MongoClient('47.74.40.167')['VnTrader_Tick_Db'][symbol]
        collection_clean = pymongo.MongoClient('localhost')['VnTrader_Tick_Db_CLEAN_QD'][symbol]

        md = VnDataclean(collection_origin, collection_clean)
        md.run(date_range)
    """

    # todo
    """
    # todo 合成level2数据（仅BITMEX？）
    # todo 标记异常数据：闪崩(价格相比前1tick/前1min相差>10%)、bug价格(okex的20190125 eth合约价格仅剩1usd)
    # todo 标记数据来源：vnpy实盘记录，1Token（方便以后数据清理时用）
    # todo 保存STATUS：插入时把30分钟时段拆分为1min插入；读取时，把30分钟时段拆分为1min，再过滤已清洗数据，再把时间再按30周期整合
    """

    STATUS_FOUND = 'FOUND'
    STATUS_UNFOUND = 'UNFOUND'
    STATUS_CLEANED = 'FINISHED'

    def __init__(self, collection_origin, collection_target, type,
                 data_status=None, bar_collection=None, isReal=False):
        self.collection_origin = collection_origin  # 原始collection
        self.collection_target = collection_target  # 目标collection
        self.type = type                            # 操作：'copy'/'clean'

        self.data_status = data_status              # 状态collection
        self.bar_collection = bar_collection        # 1min bar collection

        self.isReal = isReal

        self.date_isExist_dict = {}

    # 1.多线程启动函数
    def run(self, date_range):
        """

        :param date_range: pd.date_range
        :param isReal:
        :return:
        """
        # 真实模式（多线程运行）
        if self.isReal:
            # 创建索引
            creat_index(self.collection_origin, 'date')
            creat_index(self.collection_origin, 'time')

            # 多线程运行
            tp = ThreadPool(200)
            for startdate,enddate in date_range:
                tp.add(target=self.worker, args=(startdate, enddate, self.type))
            tp.run(isJoin=True)

            # 删除索引
            drop_index(self.collection_origin, 'date')
            drop_index(self.collection_origin, 'time')

        # 调试模式（单线程运行）
        else:
            for startdate,enddate in date_range:
                self.worker(startdate, enddate, self.type)

    # 2.执行函数
    @retry(fix_exception=(Exception), fix_func=sendMail_connectError, retry_times=2)
    def worker(self, startdate, enddate, type):
        print('{} start: {} - {}'.format(self.collection_origin._Collection__name, startdate, enddate))

        # 读取数据
        tickData_origin = self.load_data(startdate, enddate)
        if tickData_origin is None:
            return

        # 复制
        if type == 'copy':
            self.collection_target.insert(tickData_origin.to_dict('records'))

        # tick清洗
        elif type == 'clean':
            tickData_clean = self.clean_tick_data(tickData_origin)
            self.collection_target.insert(tickData_clean.to_dict('records'))

            # 合成1min bar
            if self.bar_collection:
                data_bar_1min = transform_tick_1MinBar(tickData_clean)
                self.bar_collection.insert(data_bar_1min.to_dict('records'))

        # 保存已被修改数据
        if self.isReal:
            self.save_status(startdate, enddate, self.STATUS_CLEANED)

        print('{} finish: {} - {}'.format(self.collection_origin._Collection__name, startdate, enddate))

    def load_data(self, startdate, enddate):
        startdate_date = dt.datetime.strftime(startdate, '%Y%m%d')
        enddate_date = dt.datetime.strftime(enddate, '%Y%m%d')
        startdate_time = dt.datetime.strftime(startdate, '%H:%M:%S')
        enddate_time = dt.datetime.strftime(enddate, '%H:%M:%S')

        flt = {'date': {'$gte':startdate_date, '$lte':enddate_date},
               'time': {'$gte':startdate_time, '$lt':enddate_time}}
        unuseful_field = {"_id" : 0, "rawData" : 0, 'timeZone': 0}
        cursor_origin = mongo_aggregate(collection=self.collection_origin, match=flt, project=unuseful_field, sort={'datetime': 1})

        data_origin = pd.DataFrame(list(cursor_origin))
        if not len(data_origin):
            self.save_status(startdate, enddate, self.STATUS_UNFOUND)
            print('UNFOUNED: {} {}'.format(startdate, enddate))
            return None

        return data_origin

    def clean_tick_data(self, data):
        columns = data.columns

        # 格式标准化
        col_list = ['lastPrice', 'lastVolume',
                    u'askPrice1', u'askPrice2', u'askPrice3', u'askPrice4', u'askPrice5',
                    u'askVolume1', u'askVolume2', u'askVolume3', u'askVolume4', u'askVolume5',
                    u'bidPrice1', u'bidPrice2', u'bidPrice3', u'bidPrice4', u'bidPrice5',
                    u'bidVolume1', u'bidVolume2', u'bidVolume3', u'bidVolume4', u'bidVolume5']
        for col in col_list:
            data[col] = data[col].astype(np.float64)

        # 删除全部字段重复的数据（∵读取数据时已经排除了无用字段，所以直接按“全部字段重复”来删除）
        data.drop_duplicates(subset=None, keep='first', inplace=True)

        # 删除错误数据
        cond1 = data['lastPrice'] > 0
        cond2 = data['askPrice1'] > 0
        cond3 = data['bidPrice1'] > 0

        # 只保留成交和第一档有变化的数据
        data['level2'] = data['lastPrice'] + data['lastVolume'] + data['askPrice1'] + data['bidPrice1'] + data['askVolume1'] + data['bidVolume1']
        cond4 = data['level2'] != data['level2'].shift(1)
        data = data.ix[(cond1 & cond2 & cond3 & cond4)]
        del data['level2']

        # 标记dataType
        data['dataType'] = 'depth'
        data.ix[(data['lastPrice'] != data['lastPrice'].shift(1)), 'dataType'] = 'tick'
        return data

    # 3.过滤时段
    def dateRange_filterStatus(self, startdate, enddate, freq):
        dateRange_origin = pd.date_range(startdate, enddate, freq=freq)

        # 过滤已清洗时间段
        if self.data_status:
            timeQuantum_finished = self.load_status(start=startdate, end=enddate, status={'$in': [self.STATUS_UNFOUND, self.STATUS_CLEANED]})
            timeQuantum_filter = self.filter_finish_date(startdate, enddate, freq, timeQuantum_finished)
        else:
            timeQuantum_filter = [(x,y) for x,y in zip(dateRange_origin[:-1], dateRange_origin[1:])]

        # 过滤当日无数据时间段
        timeQuantum_found = self.load_status(start=startdate, end=enddate, status=self.STATUS_FOUND)
        for i in timeQuantum_found:
            start_date_0 = i[0].strftime('%Y%m%d')
            self.date_isExist_dict[start_date_0] = True
        timeQuantum_unfound = self.load_status(start=startdate, end=enddate, status=self.STATUS_UNFOUND)
        for i in timeQuantum_unfound:
            start_date_0 = i[0].strftime('%Y%m%d')
            self.date_isExist_dict[start_date_0] = False
        timeQuantum_filter = self.filter_day_and_hour(timeQuantum_filter)

        return timeQuantum_filter

    # 3.1.过滤已清洗时段
    def filter_finish_date(self, startdate, enddate, freq, timeQuantum_finished):
        # 1.已清洗数据时间段为切割为1min
        tq_finished_1 = map(lambda x: pd.date_range(x[0], x[1]-dt.timedelta(seconds=60), freq='1T').tolist(), timeQuantum_finished) # remark ∵按[start_date, end_date)来搜索，∴status里的end_date是不含该分钟的，因此要end_date-1min
        a = []
        for i in tq_finished_1:
            a.extend(i)
        b = set(a)

        # 2.预备时间段为切割为1min
        c = set(pd.date_range(startdate, enddate, freq='1T'))

        # 3.获取尚未清洗时间段的数据
        rl = list(c.difference(b))
        rl.sort()

        # 4.切割不连续时段
        df = pd.DataFrame({'datetime': rl}, index=rl)
        df['diff'] = df['datetime'].diff()
        df_cut1 = np.where(df['diff'] > dt.timedelta(seconds=60))

        # 5.把各个不连续时段切割，并重构为freq的时间段
        if df_cut1:
            timeQuantum_filter = []
            df_cut2 = [0] + df_cut1[0].tolist() + [len(df)]
            for inx_s, inx_e in zip(df_cut2[:-1], df_cut2[1:]):
                df3 = df.iloc[inx_s:inx_e]

                df2 = pd.DataFrame()
                df2['start'] = df3['datetime'].resample(freq).first()
                df2['end'] = df3['datetime'].resample(freq).last() + dt.timedelta(seconds=60)
                df2.dropna(inplace=True)

                timeQuantum_filter.extend([(x,y) for x,y in zip(df2['start'], df2['end'])])

        else:
            df3 = df
            df2 = pd.DataFrame()
            df2['start'] = df3['datetime'].resample(freq).first()
            df2['end'] = df3['datetime'].resample(freq).last() + dt.timedelta(seconds=60)
            df2.dropna(inplace=True)

            timeQuantum_filter = [(x,y) for x,y in zip(df2['start'], df2['end'])]

        return timeQuantum_filter

    def save_status(self, start, end, status):
        if self.data_status:
            return self.data_status.save_status(start, end, status)

    def load_status(self, start, end, status=None):
        if self.data_status:
            return self.data_status.load_status(start, end, status)
        else:
            return []

    # 3.2.过滤当日无数据的时段
    def filter_day_and_hour(self, timeQuantum_filter):
        df = pd.DataFrame(timeQuantum_filter, columns={'start', 'end'})
        df['date'] = df['start'].map(lambda x: x.date())

        if self.isReal:
            creat_index(self.collection_origin, 'date')

        # 判断collection_origin在对应日期是否存在数据
        tp = ThreadPool(30)
        for date in set(df['date'].values):
            tp.add(target=self.date_isExist, args=(date.strftime('%Y%m%d'), ))
        tp.run(isJoin=True)

        df['exist'] = df['date'].map(lambda x: self.date_isExist_dict[x.strftime('%Y%m%d')])
        df = df.ix[df['exist']]
        timeQuantum_filter = [(x,y) for x,y in zip(df['start'], df['end'])]
        return timeQuantum_filter

    def date_isExist(self, date):
        date_dt = dt.datetime.strptime(date, '%Y%m%d')
        exist = self.date_isExist_dict.get(date)

        if not exist:
            doc = True if self.collection_origin.find_one({'date': date}) else False

            if self.data_status:
                if doc:
                    self.save_status(date_dt, date_dt + dt.timedelta(days=1), self.STATUS_FOUND)
                else:
                    self.save_status(date_dt, date_dt + dt.timedelta(days=1), self.STATUS_UNFOUND)

            isExist = self.date_isExist_dict[date] = doc

        else:
            isExist = exist

        print('Is Exist?: {}, {}'.format(date, isExist))
        return isExist

def run_VnTickData_Clean():
    # 设定
    # set_proxy_shadowsocks()     # 使用shadowsocks代理

    database_origin = pymongo.MongoClient('localhost')['VnTrader_Tick_Db_origin']
    database_target = pymongo.MongoClient('localhost')['VnTrader_Tick_Db_ok']
    database_target_bar = pymongo.MongoClient('localhost')['VnTrader_1Min_Db_ok']
    collection_status = pymongo .MongoClient('localhost')['VnTrader_Status_Db']['DataStatus']

    # 处理所有collection
    collection_name = database_origin.collection_names()
    collection_name.sort()
    # collection_name = collection_name[collection_name.index(u'BCH-USD-190705.OKEX'):]

    # 只对特定数据进行处理
    # collection_name = ['BTC-USD-SWAP.OKEX', 'BCH-USD-SWAP.OKEX', 'LTC-USD-SWAP.OKEX', 'ETH-USD-SWAP.OKEX', 'ETC-USD-SWAP.OKEX', 'EOS-USD-SWAP.OKEX', 'XRP-USD-SWAP.OKEX']
    # collection_name = ['btcusdt.HUOBI', 'bchusdt.HUOBI', 'ltcusdt.HUOBI', 'ethusdt.HUOBI', 'etcusdt.HUOBI', 'eosusdt.HUOBI']

    # 执行
    for symbol in collection_name:
        collection_origin = database_origin[symbol]
        collection_target = database_target[symbol]

        collection_target_bar = database_target_bar[symbol]
        data_status = DataStatus(collection_status, collection_origin, collection_target)

        vc = VnTickData_Clean(collection_origin, collection_target, type='clean',
                              data_status=data_status, bar_collection=collection_target_bar, isReal=True)
        dateRange = vc.dateRange_filterStatus('20190310 00:00', '20190903 00:00', '15T')
        vc.run(dateRange)

def run_VnTickData_Clean_wrongdata():
    # temp 特殊处理，把重复的数据取出，再插入到相同的数据里

    database_origin = pymongo.MongoClient('localhost')['VnTrader_Tick_Db_origin']
    database_target = pymongo.MongoClient('localhost')['VnTrader_Tick_Db_origin']
    collection_status = pymongo.MongoClient('localhost')['VnTrader_Status_Db']['DataStatus']

    ctnameList = database_origin.collection_names()
    ctnameList.sort()
    ctnameList2 = filter(lambda x: x if getVtSymbolExchange(x) == 'CTP' and '-' in x else None, ctnameList)

    for symbol in ctnameList2:
        collection_origin = database_origin[symbol]
        collection_target = database_target[symbol+'.OKEX']
        data_status = DataStatus(collection_status, collection_origin, collection_target)

        vc = VnTickData_Clean(collection_origin, collection_target, type='copy', data_status=data_status)
        date_range = vc.dateRange_filterStatus('20190101', '20190901', '30T')
        vc.run(date_range, isReal=True)


# 从文件夹的HDF5导入到MongoDB
def import_from_hdf5(loadpath, host='localhost', port=27017):
    """:vnpy的collectionName类型

    [IF1809, IF.HOT, EUR/USD.FXCM, BTCUSD.BITMEX, BTC-USD-190625.OKEX]
    （无'.'的CTP普通合约，'.HOT'的CTP主连，带'/'的FXCM合约，普通的vtSymbol）
    """

    client = pymongo.MongoClient(host=host, port=port)

    pathlist = getPathfiles_dirname_filename_filepath(loadpath)
    for dbname, filename, filepath in pathlist:
        # filename处理：1.'EUR%USD'→'EUR/USD' 2.'.BADAXBT.BITMEX'→'_BADAXBT.BITMEX' 3.tickData带‘#’('bkbteth.HUOBI#201812210303')
        filename = filename.replace('%','/')
        if filename[0] == '.':
            filename = filename.replace('.', '_', 1)
        if '#' in filename:
            filename = re.findall("(.+?\d*)#", filename)[0]  # remark filename.replace('%','/')处理'EUR%USD'→'EUR/USD'； "(.*?\d*)_"→"(.+?\d*)_"处理'_BADAXBT.BITMEX'
        ctname = filename.replace('.H5', '')
        print(dbname, ctname)

        data = read_hdf(filepath)
        ct = client[dbname][ctname]
        import_collection(ct, data)


"""数据下载"""
# [下载] 从okex交易所下载永续合约(swap)的1min数据
def download_okexfSwap(symbol='BTC-USD-SWAP', startdate='201908080100', enddate='201908080200', collection=None):
    """
    官方文档：https://www.okex.com/docs/zh/#swap-swap---line
    url样例：https://www.okex.com/api/swap/v3/instruments/BTC-USD-SWAP/candles?start=2019-08-08T02:00:00Z&end=2019-08-08T03:00:00Z&granularity=60

    注意：截止至2019/8/8，一次请求200条，而且只能请求近期数据，超过一定天数的数据不允许请求

    :param symbol:
    :param startdate:
    :param enddate:
    :return:
    """
    startdate = dt.datetime.strptime(startdate, "%Y%m%d%H%S").isoformat()
    enddate = dt.datetime.strptime(enddate, "%Y%m%d%H%S").isoformat()

    url = r'https://www.okex.com/api/swap/v3/instruments/{}/candles?start={}Z&end={}Z&granularity=60'.format(symbol, startdate, enddate)
    print(url)

    data = requests.get(url)

    df = pd.DataFrame(eval(data.text))
    if len(df):
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'volume_currency']

        df['vtSymbol'] = '{}.{}'.format(symbol, 'OKEX')
        df['symbol'] = symbol
        df['exchange'] = 'OKEX'

        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)

        df['datetime'] = df['datetime'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
        df['date'] = df['datetime'].map(lambda x: x.strftime('%Y%m%d'))
        df['time'] = df['datetime'].map(lambda x: x.strftime('%H:%M:%S'))

        df['volume'] = df['volume'].astype(int)
        df['openInterest'] = None
        df['interval'] = None

        df['dataSource'] = 'EXCHANGE'      # new 数据来源
        df['volume_currency'] = df['volume_currency'].astype(float)     # 独有数据：以币计价的成交量

        df = df.sort_values('datetime')

        print(df)

    if collection:
        collection.insert(df.to_dict('records'))

def run_download_okexfSwap():
    symbol = 'BTC-USD-SWAP'
    db = pymongo.MongoClient('localhost')['VnTrader_1Min_Db_test']
    ct = db[symbol+'.OKEX']
    download_okexfSwap(symbol=symbol, startdate='201908080100', enddate='201908080200', collection=ct)


# [下载] 从火币API下载火币历史数据
def download_huobi():
    # notice 运行环境要求：python3.7.1 + 服务器（翻墙） + pip install websocket_client（不能是websocket）

    # region # 导包
    import datetime as dt
    import pandas as pd
    # from vnpy.trader.vtObject import VtBarData

    from websocket import create_connection
    import pymongo, time, gzip

    pd.set_option('display.max_rows',50)
    pd.set_option('display.max_columns',100)
    pd.set_option('display.width', 1000)
    pd.set_option('expand_frame_repr', False)  # 当列太多时不自动换行
    pd.set_option('display.float_format', lambda x: '%.2f' % x)  # 设置表的长度、宽度 & 不采用科学计数法
    # endregion

    # 计算请求日期列表
    symbolist = ['btcusdt', 'bchusdt', 'ltcusdt', 'ethusdt', 'etcusdt', 'eosusdt', 'xrpusdt']

    for symbol in symbolist:
        # 锁定集合，并创建索引
        client = pymongo.MongoClient("localhost", 27017)
        collection = client['VnTrader_1Min_Db'][symbol+'.HUOBI']

        start_time = dt.datetime.strptime('20190821 00:00', '%Y%m%d %H:%M')
        end_time = dt.datetime.strptime('20190901 00:00', '%Y%m%d %H:%M')

        # 计算请求日期表
        datetimeList = pd.date_range(start_time, end_time, freq='4H').tolist()      # 每次请求数据最多不超过200条
        datetimeList.append(end_time)
        request_datetime = list(zip(datetimeList[:-1], datetimeList[1:]))

        ws = create_connection('wss://api.huobi.pro/ws')
        for start_ctime, end_ctime in request_datetime:
            end_ctime = end_ctime - dt.timedelta(seconds=60)    # 取[start_ctime, end_ctime)，即不取最后一分钟
            print(start_ctime, end_ctime)

            start_ctime = int(time.mktime(start_ctime.timetuple()))
            end_ctime = int(time.mktime(end_ctime.timetuple()))

            # 请求
            request = """{"req": "market.%s.kline.1min","id": "id10", "from": %s, "to": %s}""" % (symbol, int(start_ctime), int(end_ctime))
            ws.send(request)
            print(request)

            time.sleep(1)

            data = ws.recv()
            result = gzip.decompress(data).decode('utf-8')

            # ping-pong
            if result[:7] == '{"ping"':
                ts = result[8:21]
                pong = '{"pong":' + ts + '}'
                ws.send(pong)

                # 重新发送请求
                time.sleep(1)
                ws.send(request)
                time.sleep(1)

                data = ws.recv()
                result = gzip.decompress(data).decode('utf-8')

            dict_ = eval(result)
            if not dict_.get('data', None):
                continue

            # 格式转换
            data = pd.DataFrame(dict_['data'])

            data['vtSymbol'] = '.'.join([symbol, 'HUOBI'])
            data['symbol'] = symbol
            data['exchange'] = 'HUOBI'

            data['datetime'] = data['id'].map(dt.datetime.fromtimestamp)
            data['date'] = data['datetime'].map(lambda x: x.date().strftime('%Y%m%d'))
            data['time'] = data['datetime'].map(lambda x: x.time().strftime('%H:%M:%S'))

            data['volume'] = data['vol']
            data['tobtcvolume'] = data['amount']

            data['openInterest'] = 0
            data['interval'] = u''
            data['dataSource'] = 'EXCHANGE'

            data = data[['vtSymbol', 'symbol', 'exchange',
                         'datetime', 'date', 'time',
                         'open', 'high', 'low', 'close', 'volume', 'tobtcvolume',
                         'openInterest', 'interval', 'dataSource']]
            print(data)

            # 插入
            collection.insert(data.to_dict('records'))

            # # 更新
            # for i, bar in data.iteritems():
            #     flt = {'datetime': bar['datetime']}
            #     collection.update_one(flt, {'$set':bar}, upsert=True)


# [下载] 下载1Token数据
def download_1Token(start_time, end_time, sep, token_headers, token_contract,
                    mongo_collection, symbol, vtSymbol, exchange, gatewayName):
    """
    # 1Token变量命名规则：https://1token.trade/docs#/instruction/naming-rules

    :param start_time:
    :param end_time:
    :param sep:
    :param token_headers:
    :param token_contract:
    :param mongo_collection:
    :param symbol:
    :param vtSymbol:
    :param exchange:
    :param gatewayName:
    :return:
    """
    # 获取数据
    candle_download_url = r"https://hist-quote.1tokentrade.cn/candles?contract={contract}&since={since}&until={until}&duration={duration}&format=json"\
        .format(contract=token_contract, since=start_time, until=end_time, duration=sep)
    data_requests = requests.get(candle_download_url, headers=token_headers)   # 下载行情必须要有handers key才能下载
    data = eval(data_requests.text)

    # quota次数不够则返回
    if isinstance(data, dict):
        msg = data.get('message')
        if msg and 'no enough' in msg:
            print(msg)
            return

    # 转换为DataFrame
    df = pd.DataFrame(data)
    if not len(df):
        return

    df['datetime'] = df['timestamp'].map(dt.datetime.fromtimestamp)
    df = df[['datetime', 'open', 'close', 'high', 'low', 'volume']]

    df['date'] = df['datetime'].map(lambda x: datetime.strftime(x.date(), '%Y%m%d'))
    df['time'] = df['datetime'].map(lambda x: str(x.time()))
    df['vtSymbol'] = vtSymbol
    df['symbol'] = symbol
    df['exchange'] = exchange
    df['gatewayName'] = gatewayName

    df['dataSource'] = '1Token'
    df['interval'] = u''
    df['openInterest'] = 0

    mongo_collection.insert(df.to_dict('records'))
    print(df)

def run_download_1Token():
    headers = {"ot-key": "jbHa92JV-gEaBy0Ha-Ugx7VK1e-Q7Xll4T0"}
    # headers = {"ot-key": "SkMSYgCB-9E1MDT9q-ct1moBjV-wgJEp1IS"}
    # headers = {"ot-key": "lvaMvjVP-mXo5ZCx8-ZqeMqe5a-0KtNI6N9"}

    date_range = pd.date_range('2019-09-18', '2019-09-20').strftime('%Y-%m-%d')
    contractList = ['okswap/btc.usd.td', 'okswap/bch.usd.td', 'okswap/ltc.usd.td', 'okswap/eth.usd.td', 'okswap/etc.usd.td', 'okswap/eos.usd.td']
    vtSymbolList = ['BTC-USD-SWAP.OKEX', 'BCH-USD-SWAP.OKEX', 'LTC-USD-SWAP.OKEX', 'ETH-USD-SWAP.OKEX', 'ETC-USD-SWAP.OKEX', 'EOS-USD-SWAP.OKEX']
    gatewayName = 'OKEXF'

    for contract, vtSymbol in zip(contractList, vtSymbolList):
        symbol, exchange = vtSymbol.split('.')
        collection = pymongo.MongoClient('localhost')['VnTrader_1Min_Db_CLEAN'][vtSymbol]

        tp = ThreadPool(1)    # 设为1时为单线程，大于1时为多线程
        for start_time, end_time in zip(date_range[:-1], date_range[1:]):
            tp.add(download_1Token,
                  kwargs = dict(start_time=start_time, end_time=end_time, sep='1m',
                            token_headers=headers, token_contract=contract,
                            mongo_collection=collection,
                            symbol=symbol, vtSymbol=vtSymbol, exchange=exchange, gatewayName=gatewayName))
        tp.run(isJoin=True, sleep=3)


