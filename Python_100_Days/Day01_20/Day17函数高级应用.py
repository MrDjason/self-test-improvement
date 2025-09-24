# Day 17 函数高级应用

# 装饰器
# 装饰器是“用一个函数装饰另外一个函数并为其提供额外的能力”的语法现象
# 装饰器本身是一个函数，它的参数是被装饰的函数，它的返回值是一个带有装饰功能的函数。

# 举例
import random
import time

def download(filename):
    """下载文件"""
    print(f'开始下载{filename}.')
    time.sleep(random.random() * 6)
    # time.sleep(seconds) 暂停线程seconds秒
    # random.random() * 6 其中random.random()生成一个0.0-1.0的浮点数，这里要生成0.0-6.0的浮点数
    print(f'{filename}下载完成.')

    
def upload(filename):
    """上传文件"""
    print(f'开始上传{filename}.')
    time.sleep(random.random() * 8)
    print(f'{filename}上传完成.')

"""    
download('MySQL从删库到跑路.avi')
upload('Python从入门到住院.pdf')

start = time.time()
# time.time() 返回当前时间戳
download('MySQL从删库到跑路.avi')
end = time.time()
print(f'花费时间: {end - start:.2f}秒')
start = time.time()
upload('Python从入门到住院.pdf')
end = time.time()
print(f'花费时间: {end - start:.2f}秒')
# 重复过多，有简单的操作吗？
"""

def record_time(func):
    
    def wrapper(*args, **kwargs):
        
        result = func(*args, **kwargs)
        
        return result
    
    return wrapper



def record_time(func):

    def wrapper(*args, **kwargs):
        # 在执行被装饰的函数之前记录开始时间
        start = time.time()
        # 执行被装饰的函数并获取返回值
        result = func(*args, **kwargs)
        # 在执行被装饰的函数之后记录结束时间
        end = time.time()
        # 计算和显示被装饰函数的执行时间
        print(f'{func.__name__}执行时间: {end - start:.2f}秒')
        # 返回被装饰函数的返回值
        return result
    
    return wrapper
"""
download = record_time(download)
# 用record_time装饰download函数,运行过程中增加计时功能
upload = record_time(upload)
# 用record_time装饰upload函数
download('MySQL从删库到跑路.avi')
upload('Python从入门到住院.pdf')
"""
import random
import time


def record_time(func):

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__}执行时间: {end - start:.2f}秒')
        return result

    return wrapper


@record_time
def download(filename):
    print(f'开始下载{filename}.')
    time.sleep(random.random() * 6)
    print(f'{filename}下载完成.')


@record_time
def upload(filename):
    print(f'开始上传{filename}.')
    time.sleep(random.random() * 8)
    print(f'{filename}上传完成.')

"""
download('MySQL从删库到跑路.avi')
upload('Python从入门到住院.pdf')
"""

# 模板
"""
def main(func):
    def second(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return second
"""



# 如果想去掉装饰器的作用执行原函数
from functools import wraps
# Python 标准库functools模块的wraps函数也是一个装饰器
# 我们将它放在wrapper函数上，可以帮我们保留被装饰之前的函数
# 可以通过被装饰函数的__wrapped__属性获得被装饰之前的函数。

def record_time(func):

    @wraps(func) # 重要
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__}执行时间: {end - start:.2f}秒')
        return result

    return wrapper


@record_time
def download(filename):
    print(f'开始下载{filename}.')
    time.sleep(random.random() * 6)
    print(f'{filename}下载完成.')


@record_time
def upload(filename):
    print(f'开始上传{filename}.')
    time.sleep(random.random() * 8)
    print(f'{filename}上传完成.')
"""
# 调用装饰后的函数会记录执行时间
download('MySQL从删库到跑路.avi')
upload('Python从入门到住院.pdf')
# 取消装饰器的作用不记录执行时间
download.__wrapped__('MySQL必知必会.pdf')
upload.__wrapped__('Python从新手到大师.pdf')
"""
def main(func):
    
    @wraps(func)
    def second(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return second
# xxx.__wrapped__(' ')

# 递归调用
# 函数直接 / 间接调用自身就是递归
# 注意 1. 收敛条件 2. 递推公式

# 阶乘函数
def fac(num):
    if num in (0, 1):       # 收敛条件 等价于 if num == 0 or num == 1:
    # (0, 1) 为元组...
        return 1
    return num * fac(num-1) # 递推公式 、
# 递归调用函数入栈
# 5 * fac(4)
# 5 * (4 * fac(3))
# 5 * (4 * (3 * fac(2)))
# 5 * (4 * (3 * (2 * fac(1))))
# 停止递归函数出栈
# 5 * (4 * (3 * (2 * 1))) = 120

def fib1(n):
    if n in (1, 2):               # 收敛条件：前两项为1
        return 1
    return fib1(n-1) + fib1(n-2)  # 递归公式

for i in range(1, 21):
    print(fib1(i))

# fib1(5) = fib1(4) + fib1(3) → 3 + 2 = 5
# fib1(4) = fib1(3) + fib1(2) → 2 + 1 = 3
# fib1(3) = fib1(2) + fib1(1) → 1 + 1 = 2
# ...