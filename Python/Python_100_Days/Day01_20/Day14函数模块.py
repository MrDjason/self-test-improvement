# Day 14 函数模块

# 计算组合数 C(m,n)
"""
m = int(input('请输入m：'))
n = int(input('请输入n：'))
fm = 1   
#计算 m 的阶乘
for i in range(1, m + 1):
    fm *= i

fn = 1
# 计算 n 的阶乘
for i in range(1, n + 1):
    fn *= i

fk = 1
# 计算 m-n 的阶乘
for i in range(1 , m - n + 1):
    fk *= i

print(fm // fn // fk)
"""

# 我们可以将求阶乘的功能封装到一个称为“函数”的代码块中
# 在需要计算阶乘的地方，我们只需“调用函数”即可实现对求阶乘功能的复用
"""
def fac(num):
    result = 1
    for i in range(1,num+1):
        result *= i
    return result
m = int(input('m = '))
n = int(input('n = '))

print(fac(m) // fac(n) // fac(m - n))
"""

from math import factorial # from math import factorial as f 定义为f更简单！

"""
m = int(input('m = '))
n = int(input('n = '))
print(factorial(m) // factorial(n) // factorial(m - n))
"""
# 位置参数和关键字参数

def make_judgement(a, b, c):
    """判断三条边的长度能否构成三角形"""
    return a + b > c and b + c > a and a + c > b

print(make_judgement(1,2,3)) # False
print(make_judgement(4,5,6)) # True
# 也可以不按位置排
print(make_judgement(b = 2,  c= 2, a = 2)) # True

# /设置强制位置参数
def make_judgement(a, b, c, /):
    """判断三条边的长度能否构成三角形"""
    return a + b > c and b + c > a and a + c > b
try:
    print(make_judgement(b=2, c=3, a=1))
except TypeError as e: 
    print(f'错误：{e}')

# 用*设置命名关键字参数
def make_judgement(*, a, b, c):
    """判断三条边的长度能否构成三角形"""
    return a + b > c and b + c > a and a + c > b
try:
    print(make_judgement(1, 2, 3))
except TypeError as e: 
    print(f'错误：{e}')

# 参数的默认值

def test(num = 2):
    return num

# 没有指定参数，那么num使用默认值2
print(test())
# 传入参数3，变量n被赋值为3
print(test(3))

# 可变参数
# 用星号表达式来表示args可以接收0个或任意多个参数
# 调用函数时传入的n个参数会组装成一个n元组赋给args
# 如果一个参数都没有传入，那么args会是一个空元组

def add(*args):
    total = 0
    for var in args:
        if type(var) in (int, float):
            total += var
    return total

# 调用时可以传入0个或多个参数
print(add())                      # 0
print(add(1, 2, 3, 4))            # 10
print(add(1, 2, 'Hello', 3, 4.5)) # 10.5

# 以“参数名=参数值”的形式传入若干个参数，可以通过**kwargs传入字典
def foo(*args, **kwargs):
    print(args)
    print(kwargs)
foo(3, 2.1, True, name='骆昊', age=43, gpa=4.95)

# 通过*args接收多个参数，形成元组。通过**kwargs接收多个“x=y”形成字典

def foo():
    print('hello, world!')

def foo():
    print('goodbye, world!')

foo()  # 调用foo函数会输出最后一个函数/感觉相当于重新定义了函数



"""
module1.py
def foo():
    print('hello, world!')

module2.py
def foo():
    print('goodbye, world!')
test.py

import module1
import module2

# 用“模块名.函数名”的方式（完全限定名）调用函数，
module1.foo()  # hello, world!
module2.foo()  # goodbye, world!

# 还可以用as作为别名
import module1 as m1
import module2 as m2

m1.foo()  # hello, world!
m2.foo()  # goodbye, world!
"""

"""
Python内置函数库
abs	返回一个数的绝对值，例如：abs(-1.3)会返回1.3。
bin	把一个整数转换成以'0b'开头的二进制字符串，例如：bin(123)会返回'0b1111011'。
chr	将Unicode编码转换成对应的字符，例如：chr(8364)会返回'€'。
hex	将一个整数转换成以'0x'开头的十六进制字符串，例如：hex(123)会返回'0x7b'。
input	从输入中读取一行，返回读到的字符串。
len	获取字符串、列表等的长度。
max	返回多个参数或一个可迭代对象中的最大值，例如：max(12, 95, 37)会返回95。
min	返回多个参数或一个可迭代对象中的最小值，例如：min(12, 95, 37)会返回12。
oct	把一个整数转换成以'0o'开头的八进制字符串，例如：oct(123)会返回'0o173'。
open	打开一个文件并返回文件对象。
ord	将字符转换成对应的Unicode编码，例如：ord('€')会返回8364。
pow	求幂运算，例如：pow(2, 3)会返回8；pow(2, 0.5)会返回1.4142135623730951。
print	打印输出。
range	构造一个范围序列，例如：range(100)会产生0到99的整数序列。
round	按照指定的精度对数值进行四舍五入，例如：round(1.23456, 4)会返回1.2346。
sum	对一个序列中的项从左到右进行求和运算，例如：sum(range(1, 101))会返回5050。
type	返回对象的类型，例如：type(10)会返回int；而 type('hello')会返回str。
"""