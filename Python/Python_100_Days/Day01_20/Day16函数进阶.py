# Day 16 函数进阶

# 高阶函数
def calc(*args, **kwargs): # 多个参数求和
    items = list(args) + list(kwargs.values())
    result = 0
    for item in items:
        if type(item) in (float, int):
            result += item
    return result

# 自定义运算规则进行二元运算
def calc(first_value, function, *args, **kwargs):
    items = list(args) + list(kwargs.values())
    result = first_value
    for item in items:
        if type(item) in (float, int):
            result = function(result, item)
    return result

def addtion(x, y):
    return x+y
def subtraction(x, y):
    return x-y
def multiplication(x, y):
    return x*y
def division(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return "错误：除数不能为 0"
    except TypeError:
        return "错误：除数和被除数必须是数值类型"
    
# 1. 测试加法
print("加法测试（1+2+3+4+5）：", calc(0, addtion, 1, 2, 3, a=4, b=5)) # 15
# 2. 测试减法（
print("减法测试（0-5-3-2）：", calc(0, subtraction, 5, 3, a=2))       # 4
# 3. 测试乘法
print("乘法测试（0*2*3*4）：", calc(0, multiplication, 2, 3, a=4))    # 0
# 4. 测试除法
print("除法测试（0/2/1）：", calc(0, division, 2, a=1))               # 0.0

# 引用库
import operator

print(calc(0, operator.add, 1, 2, 3, 4, 5))  # 15
print(calc(1, operator.mul, 1, 2, 3, 4, 5))  # 120

def is_even(num):
    """判断num是不是偶数"""
    return num % 2 == 0

def square(num):
    """求平方"""
    return num ** 2

old_nums = [35, 12, 8, 99, 60, 52]
new_nums = list(map(square, filter(is_even, old_nums)))
# filter(is_even, old_nums)filter 会遍历 old_nums 中的每个元素，将元素传给 is_even 函数
# 如果 is_even(元素) 返回 True，则保留该元素；否则剔除。
# map(square, filter(...)) 将过滤后的元素经过square处理，再形成一个map类型迭代
# list(map(square,filter(...))) 将map类型迭代转化为列表
print(new_nums)  # [144, 64, 3600, 2704]

new_nums = [i**2 for i in old_nums if i % 2 ==0]
print(new_nums)

# sorted 排序函数
old_strings = ['in', 'apple', 'zoo', 'waxberry', 'pear']
new_strings = sorted(old_strings)
print(new_strings)  # ['apple', 'in', 'pear', waxberry', 'zoo']

# lambda函数
# lambda 函数的本质是 “简化简单逻辑的函数定义”，不需要写 def、函数名和 return
# 格式：lambda 参数（输入）: 表达式（输出）
add_function = lambda x,y:x+y
print(add_function(1,2)) # 3

# 代替简单函数
old_nums = [35, 12, 8, 99, 60, 52]
new_nums = list(map(lambda x:x**2,filter(lambda x:x%2==0,old_nums)))
# lambda x:x%2==0 传入一个x，若是偶数返回True
print(new_nums)  # [144, 64, 3600, 2704]

import functools
import operator

# 用一行代码实现计算阶乘的函数
fac = lambda n: functools.reduce(operator.mul, range(2, n + 1), 1)
# functools.reduce(...)：reduce 函数的作用是 “累积运算”—— 对序列中的元素逐个应用函数，最终得到一个结果
# operator.mul 累计运算的规则【这里是乘法运算】
# range(2, n + 1)：要参与运算的序列【这里是生成2到n的序列】

# 用一行代码实现判断素数的函数
is_prime = lambda x: all(map(lambda f: x % f, range(2, int(x ** 0.5) + 1)))
# lambda x:... 传入参数x，例如传入37
# all(...)：all 函数判断 “序列中所有元素是否都为 True”，这里用map序列来判断
# range(2, int(x ** 0.5) + 1)：生成需要检查的除数范围。【2到√x】
# f的值是range(2,...)中的每个元素
# map 函数会把第一个参数（lambda 函数）依次应用到第二个参数（range 序列）的每个元素上。
# 计算 x % f ,如果 x 能被 f 整除，余数为 0（等价于布尔值 False）；如果不能被整除，余数非 0（等价于布尔值 True）

# 调用Lambda函数
print(fac(6))        # 720
print(is_prime(37))  # True

# 偏函数
# 偏函数是指固定函数的某些参数，生成一个新的函数，这样就无需在每次调用函数时都传递相同的参数
# 可以使用functools模块的partial函数来创建偏函数

# int函数在默认情况下可以将字符串视为十进制整数进行类型转换
# 如果我们修修改它的base参数，就可以定义出三个新函数，分别用于将二进制、八进制、十六进制字符串转换为整数

int2 = functools.partial(int,base = 2) # 二进制转换
int8 = functools.partial(int,base = 8) # 八进制转换
int16 = functools.partial(int,base = 16) # 十六进制转换
print(int('1001')) # 1001

print(int2('1001'))   # 9
print(int8('1001'))   # 513
print(int16('1001'))  # 4097
# partial函数的第一个参数和返回值都是函数，它将传入的函数处理成一个新的函数返回
# 通过构造偏函数，我们可以结合实际的使用场景将原函数变成使用起来更为便捷的新函数