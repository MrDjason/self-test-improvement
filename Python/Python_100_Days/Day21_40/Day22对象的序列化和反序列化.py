# Day 22 对象的序列化和反序列化
# ==================== 基本信息 ====================
'''
如何将列表或者字典数据保存到文件中？
可以将程序中的数据以JSON格式进行保存

使用JSON是因为它结构紧凑而且是纯文本，任何操作系统和编程语言都能处理纯文本
目前JSON基本上已经取代了XML（可扩展标记语言）作为异构系统间交换数据的事实标准
'''

'''
{
    name: "john",
    age: 40,
    friends: ['kevin', 'Ted'],
    cars :[
    {brand:'HUAWEI', speed:'240'},
    {brand:'BYD', speed:'250'},
    {brand:'XiaoMi', speed:'260'}
    ]
}
'''
# ==================== 导入模块 ====================
import json
# ==================== 主 程 序 ====================
my_dict = {
    'name': "john",
    'age': 40,
    'friends': ['kevin', 'Ted'],
    'cars' :[
    {'brand':'HUAWEI', 'speed':'240'},
    {'brand':'BYD', 'speed':'250'},
    {'brand':'XiaoMi', 'speed':'260'}
    ]
}
with open('data.json', 'w') as file:
    json.dump(my_dict, file)
# 将 Python 字典转换为 JSON 格式的字符串，并打印出来
'''
json模块有四个比较重要的函数，分别是：

dump - 将Python对象按照JSON格式序列化到文件中
dumps - 将Python对象处理成JSON格式的字符串
load - 将文件中的JSON数据反序列化成对象
loads - 将字符串的内容反序列化成Python对象

“序列化在计算机科学的数据处理中，是指将数据结构或对象状态转换为可以存储或传输的形式，这样在需要的时候能够恢复到原先的状态
而且通过序列化的数据重新获取字节时，可以利用这些字节来产生原始对象的副本（拷贝）。
与这个过程相反的动作，即从一系列字节中提取数据结构的操作，就是反序列化
'''
with open('data.json', 'r') as file:
# 简单还原json为文件
    my_dict = json.load(file)
    print(type(my_dict))
    print(my_dict)

'''
Python标准库中的json模块在数据序列化和反序列化时性能并不是非常理想
为了解决这个问题，可以使用三方库ujson来替换json
pip install ujson
可以通过pip search命令根据名字查找需要的三方库
可以通过pip list命令来查看已经安装过的三方库
果想更新某个三方库，可以使用pip install -U或pip install --upgrade
如果要删除某个三方库，可以使用pip uninstall命令
'''

# 使用网络API获取数据
'''
信息→网络API→基于HTTPS提供JSON格式数据→通过PYTHON发送HTTP请求给指定URL（网络API）→成功→返回HTTP响应→提供给我们JSON数据
'''
# HTTP协议介绍 https://www.ruanyifeng.com/blog/2016/08/http.html 
