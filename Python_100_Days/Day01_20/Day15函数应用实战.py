# Day 15 函数应用实战

# 随机验证码

import random
import string
# 设计一个生成随机验证码的函数，验证码由数字和英文大小写字母构成，长度可以通过参数设置。
# 
ALL_CHARTS = string.digits + string.ascii_letters
# 所有字符=数字+大小写字母 
# string.digits = '0123456789'
# string.ascii_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def generate_code(*, code_len=4): # 通过 * 强制关键字参数传入
    """
    :params code_len:验证码的长度(默认值4字符)
    :return:由数字和英文大小写字母构成的随机验证码
    """
    return ''.join(random.choices(ALL_CHARTS, k=code_len))

"""
抽样机制：
random.sample(population, k)  # 从总体序列中随机抽取k个不重复的元素
random.choices(population, k) # 从总体序列中随机抽取k个可以重复的元素
"""
for _ in range(5):
    print(generate_code(code_len=10))

# 判断素数
def is_prime(num:int) -> bool:
    """
    :params num:int: 类型提示,需要判断的数字应该为 int 类型
    :-> bool: 类型提示,用于说明函数的返回值类型是布尔值(bool)
    :return: True表示素数,False表示非素数
    """
    # 素数是大于 1 的正整数，且仅能被 1 和自身整除
    """
    if num <= 1:
        return False
    for i in range(2,num):
        if num % i == 0 and i != num:
            return False
    return True 
    数值太大则需要循环很多次
    """
    # 如果num不是素数，那么它一定有一个因子≤√num（num的平方根）
    # a ≤ b，所以a × a ≤ a × b = num,即a² ≤ num，因此a ≤ √num
    if num <= 1:
        return False
    for i in range(2,int((num ** 0.5)+1)):
        if num % i == 0:
            return False
    return True
        
print(is_prime(13))  # True
print(is_prime(14))  # False

# 最大公约数和最小公倍数
# x 和 y 的最大公约数是能够同时整除 x 和 y 的最大整数
# x和 y 的最小公倍数是能够同时被 x 和 y 整除的最小正整数
# gcd 和lcm满足 gcd(x,y) * lcm(x,y) = x * y
# gcd = x * y / lcm
def greatest_common_divisor(x:int, y:int) -> int:
    # 循环判断 y 除以 x 的余数是否为 0，若不为 0 则更新 x 为该余数、y 为原来的 x
    # 直到余数为 0，此时的 x 就是 x 和 y 的最大公因数
    while y % x != 0:
        x, y = y % x, x # 交换赋值
        # x = 18, y = 24 -> x = 24 % 18 = 6, y = 18
        # x = 24, y = 18 -> x = 18 % 24 = 18, y = 24
    return int(x) # 返回最小公倍数

def least_common_multiple(x:int, y:int) -> int:
    # :params x:int,y:int: 需要计算最大公约数的两个整数
    return x * y //greatest_common_divisor(x, y)



print(greatest_common_divisor(18, 24)) # 6
print(least_common_multiple(18, 24))   # 72

# 双色球随机选号
#  33 个红球中选 6 个、16 个蓝球中选 1 个组成一注
RED_BALLS = [i for i in range(1,34)]  # 生成一个有1-33元素的列表
BLUE_BALLS = [i for i in range(1,17)] # 生成一个有1-16元素的列表

def choose():
    """
    随机生成一注双色球
    :return: 保存随机号码的列表
    """
    selected_balls = []
    selected_balls = random.sample(RED_BALLS, 6) # 从红球中随机抽取6个不重复的元素
    selected_balls.sort() # 对红球进行排序
    selected_balls.append(random.choice(BLUE_BALLS)) # 从蓝球中随机抽取1个元素
    return selected_balls

def display(balls):
    """
    格式输出一组号码
    :param balls: 保存随机号码的列表
    """
    print('红球：', end='')
    for ball in balls[:-1]:
        print(f'{ball:0>2d}', end=' ')
    print('蓝球：'+f'{balls[-1]:0>2d}') # 输出蓝球

# n = int(input('生成几注号码: '))
for _ in range(4):
    display(choose())