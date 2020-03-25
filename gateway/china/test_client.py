from __future__ import print_function
from __future__ import absolute_import

import threading
from time import sleep

import zmq

from vnpy.rpc import RpcClient


def get_public_network_ip():
    """获取本机公网IP"""
    from bs4 import BeautifulSoup
    from urllib.request import urlopen
    html = urlopen(r'http://ip.42.pl/raw')
    soup = BeautifulSoup(html.read(), 'html5lib')
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
    host_name = socket.gethostname()  # 获取本机名

    public_network_ip = get_public_network_ip()  # 获取公网ip

    # 设置安全组
    client = AcsClient('LTAIazo9xOTgCuJt', '6esQuEd39x3y5rMICbA3HHEgGXHAtw', 'cn-hongkong')
    request = AuthorizeSecurityGroupRequest()
    request.set_accept_format('json')
    request.set_SecurityGroupId("sg-j6c6pxfkn8n418k30v3f")
    request.set_IpProtocol("tcp")
    request.set_PortRange("27017/27017")  # 要设置的端口 格式 开始端口/结束端口
    request.set_SourceCidrIp(public_network_ip)  # 安全组-授权IP对象
    request.set_Policy("accept")
    request.set_Description(host_name)  # 安全组-描述信息
    request.set_Priority("1")

    response = client.do_action_with_exception(request)
    print(str(response, encoding='utf-8'))


class TestClient(RpcClient):
    """
    Test RpcClient   
    """

    def __init__(self, q=None):
        """
        Constructor
        """
        super(TestClient, self).__init__()
        self.__active = False
        self.bar_queue = q
        self.__socket_sub = zmq.Context().socket(zmq.SUB)
        self.set_aliyun_security = False

    def start(self, req_address: str, sub_address: str):
        if self.__active:
            return
        self.__active = True
        self.__socket_sub.connect(sub_address)
        self.__thread = threading.Thread(target=self.run)
        self.__thread.start()

    def run(self):
        """Receive data from subscribe socket"""
        while self.__active:
            topic, data = self.__socket_sub.recv_pyobj()
            self.callback(topic, data)
    def subscribe_topic(self, topic: str):
        """
        Subscribe data
        """
        self.__socket_sub.setsockopt_string(zmq.SUBSCRIBE, topic)
    def callback(self, topic, data):
        """
        Realize callable function
        """
        print(topic, data)
        if topic == "bar_data" and self.bar_queue:
            self.bar_queue.put(data)

    def _on_unexpected_disconnected(self):
        if not self.set_aliyun_security:
            set_aliyun_security_group()
            self.set_aliyun_security = True
        else:
            print("RpcServer has no response over 3 seconds, please check you connection.")


if __name__ == '__main__':
    rep_address = "tcp://*:2014"
    pub_address = "tcp://localhost:4102"
    client = TestClient()
    client.subscribe_topic("")
    client.start(rep_address,pub_address)
    # req_address = "tcp://47.26.102.183:2014"
    # sub_address = "tcp://47.56.102.183:4102"
    #
    # tc = TestClient()
    # tc.subscribe_topic("")
    # tc.start(req_address, sub_address)
    #
    # while 1:
    #     # print(tc.add(1, 3))
    #     sleep(2)
