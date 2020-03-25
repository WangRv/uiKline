# coding: utf-8
import traceback
from enum import Enum
from typing import Union

from pymongo import MongoClient, ASCENDING, ReadPreference
from pymongo.cursor import Cursor
from pymongo.errors import ConnectionFailure, OperationFailure, DuplicateKeyError, InvalidDocument
import time

from vnpy.utor.utFunction import set_aliyun_security_group, retry, UtorEmailEngine


class VnMongo():
    # email_engine = UtorEmailEngine()
    # email_engine.start()

    def __init__(self, host=None, port=27017):
        self.host = host
        self.port = port

        self.db_client = None
        self.set_aliyun_security_group_times = 0

        self.db_connect()

    def db_connect(self):
        if self.db_client:
            return

        try:
            self.db_client = MongoClient(self.host, self.port, serverSelectionTimeoutMS=1000, connect=False)

            # 调用server_info查询服务器状态，防止服务器异常并未连接成功
            self.db_client.server_info()

            # [ADD] 连接成功前等待
            while not self.db_client:
                time.sleep(1)

        except ConnectionFailure:
            if self.set_aliyun_security_group_times:   # new 如果没设置阿里云安全组，设置后重连一次
                self.db_client = None
                # set_aliyun_security_group()
                self.set_aliyun_security_group_times -= 1
                time.sleep(5)
                self.db_connect()
            else:
                self.db_client = None
                print('MongoDB连接失败')
                self.send_email('MongoDB连接失败')

    @retry(retry_exception=Exception, max_times=6, delay=5)
    def db_query(self, db_name, collection_name, flt, sort_key='', sort_direction=ASCENDING, project=None, return_cursor=False) -> Union[Cursor, list]:
        try:
            db = self.db_client[db_name]
            collection = db[collection_name]

            if not project:
                cursor = collection.find(flt)
            else:
                cursor = collection.find(flt, project)

            if sort_key:
                cursor = cursor.sort(sort_key, sort_direction)

            if return_cursor:
                return cursor
            else:
                if cursor:
                    return list(cursor)
                else:
                    return []

        except OperationFailure:
            return self.db_query_aggregated(db_name, collection_name, flt, sort_key, sort_direction, project=project, return_cursor=return_cursor)

    @retry(retry_exception=Exception, max_times=6, delay=5)
    def db_query_aggregated(self, db_name, collection_name, flt, sort_key='', sort_direction=ASCENDING, project=None, return_cursor=True, max_time_minute=10) -> Union[Cursor, list]:
        if not self.db_client:
            self.db_connect()

        db = self.db_client[db_name]
        collection = db[collection_name]

        pipeline = []
        if flt:
            pipeline.append({"$match": flt})
        if project:
            pipeline.append({"$project": project})
        if sort_key:
            pipeline.append({"$sort": {sort_key: sort_direction}})

        cursor = collection.aggregate(pipeline, allowDiskUse=True, maxTimeMS=60000*max_time_minute) # maxTimeMS单位是毫秒，60000毫秒=1分钟

        if return_cursor:
            return cursor
        else:
            if cursor.alive:
                return list(cursor)
            else:
                return []

    @retry(retry_exception=Exception, max_times=6, delay=5)
    def db_insert(self, db_name, collection_name, data, is_dataframe=False):
        if is_dataframe:
            data = data.to_dict("records")

        try:
            db = self.db_client[db_name]
            collection = db[collection_name]
            collection.insert(data)

        except DuplicateKeyError:       # '_id'重复
            db = self.db_client[db_name]
            collection = db[collection_name]
            collection.update({'_id': data['_id']}, {"$set": data}, multi=True, upsert=True)

            print('Warning：')           # [REMARK] 虽然处理了错误，但是仍然需要报出警告，使得在开发过程中立刻发现问题
            traceback.print_exc()

        except InvalidDocument:         # 无法编码为bson
            for k,v in data.items():
                if isinstance(v, Enum):
                    data[k] = v.value

            db = self.db_client[db_name]
            collection = db[collection_name]
            collection.insert(data)

            print('Warning：')           # [REMARK] 虽然处理了错误，但是仍然需要报出警告，使得在开发过程中立刻发现问题
            traceback.print_exc()

        except Exception as e:
            print('-' * 30)
            print('数据插入错误: db_name: {}, collection_name: {}'.format(db_name, collection_name))
            print('数据内容: \n{}'.format(data))
            print('错误信息: ')
            traceback.print_exc()
            print('-' * 30)

    @retry(retry_exception=Exception, max_times=6, delay=5)
    def db_update(self, db_name, collection_name, flt: dict, data: dict, upsert=True, multi=False):
        """向MongoDB中更新数据

        :param db_name:
        :param collection_name:
        :param flt: 过滤条件。
        :param data: dict类型，操作符或数据。
        :param upsert: 如果没有符合条件的数据，True表示继续插入data，False表示不插入数据。
        :param multi: 如果有多条符合条件的数据，True表示对全部符合条件的数据进行操作，False表示只对第一条符合条件的数据进行操作。
        :return:

        原子级操作：
        vm.db_update(DbName.DAILY_DB_NAME.value, ct_name, flt={}, data={"$set": {"interval": "d"}}, upsert=False, multi=True)   # 修改interval字段的值为"d"
        vm.db_update(DbName.DAILY_DB_NAME.value, ct_name, flt={}, data={"$unset": {"interval": ""}}, upsert=False, multi=True)  # 删除interval字段
        """
        db = self.db_client[db_name]
        collection = db[collection_name]
        collection.update(flt, data, upsert=upsert, multi=multi)

    @retry(retry_exception=Exception, max_times=6, delay=5)
    def db_delete(self, db_name, collection_name, flt):
        """从数据库中删除数据，flt是过滤条件"""
        db = self.db_client[db_name]
        collection = db[collection_name]
        return collection.remove(flt)

    def get_database_collection_names(self, database_name: str):
        """获取database下的collection_name列表"""
        collection_name_list = self.db_client[database_name].collection_names()
        collection_name_list.sort()
        return collection_name_list

    def get_collection(self, database_name: str, collection: str):
        """获取collection"""
        return self.db_client[database_name][collection]

    def creat_collection_index(self, db_name, collection_name, index_key, sortDirection=1):
        """创建索引"""
        db = self.db_client[db_name]
        collection = db[collection_name]

        # 如果index_key不在collection，创建index
        indexs_now = collection.index_information().keys()
        if '{}_{}'.format(index_key, sortDirection) in indexs_now:
            collection.create_index([(index_key, sortDirection)])

    def drop_collection_index(self, db_name, collection_name, index_key):
        """删除索引"""
        db = self.db_client[db_name]
        collection = db[collection_name]

        indexs_now = collection.index_information().keys()

        if '{}_1'.format(index_key) in indexs_now:
            collection.drop_index('{}_1'.format(index_key))

        elif '{}_-1'.format(index_key) in indexs_now:
            collection.drop_index('{}_-1'.format(index_key))

    @classmethod
    def send_email(cls, subject: str, content: str = "", receiver: str = "", file_path: str = ""):
        cls.email_engine.send_email(subject, content, receiver, file_path)
