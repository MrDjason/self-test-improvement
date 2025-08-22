# Day 01 下载并安装Python
# Day 02 第一个Python程序

# 快捷操作： ctrl+/ 单/多行用#注释
print("Hello World") 
# 单双引号区别
print('It\'s a book')
print("It's a book")

"""
多
行
注
释
"""

# Day 03 变量
print(0b100)  # 二进制整数   (100)B=4
print(0o100)  # 八进制整数   (100)O=64
print(100)    # 十进制整数   (100)D=100
print(0x100)  # 十六进制整数 (100)H=256
print(123.456)    # 数学写法
print(1.23456e-1)  # 科学计数法 乘10^-1次方

a, b, c, d = 100, 1.23, '100', True
print(type(a))  # <class 'int'>
print(type(b))  # <class 'float'>
print(type(c))  # <class 'str'>
print(type(d))  # <class 'bool'>

# Day 04 运算符
print(3**3, 10%3, 25//2) # 次方、取余、整除
print((a := 10))  # 海象运算符——定义、赋值、并打印

"""
f = float(input('请输入华氏温度: '))
c = (f - 32) / 1.8
print('%.1f华氏度 = %.1f摄氏度' % (f, c)) # %是占位符，%.1f是格式说明符。中间的%是格式化运算符，将右侧元组包含的变量填入左侧模板
print(f'{f:.1f}华氏度 = {c:.1f}摄氏度') # f-string，{}可以直接填写变量和表达式
"""
# Day 05 分支结构
# 赋值表达式格式化 同时输出「变量名」和「变量值」
bmi = 22.5
print(f'bmi = {bmi:.1f}')  # 输出：bmi = 22.2
print(f'{bmi = :.1f}')
"""
# match和case构造分支结构
status_code = int(input("请输入错误代码："))
match status_code:
    case 400: description = 'Bad Request'
    case 401: description = 'Unauthorized'
print('状态码描述：', description)
"""

# Day 06 循环结构
import time
for _ in range(3): # 不需要循环变量用_代替
    print(_)
    # time.sleep(1)

print(sum(range(2,101,2))) # 100内偶数相加

sum = 0
for i in range(101):
    sum += i
print(sum)

sum=0
i=1
while i<=100:
    sum+=i
    i+=1
print(sum)

# 九九乘法表
for i in range(1,10):
    for j in range(1,i+1):
        print(f'{i} * {j} = {i * j}', end = '\t') # print()结束后会换行打印 end='\t'用制表符替换换行符
    print() # 换行

# 素数判断
"""
num = int(input('请输入一个正整数判断是否为素数：'))
is_prime = True
for i in range(2,num):
    if num % i ==0:
        is_prime = False
        break
if is_prime:
    print(num, '这个数是素数')
else:
    print(num, '这个数不是素数')
"""
# Day 07 分支和循环结构实战

# 100以内素数
prime_list =[]
for i in range(2,101):
    is_prime = True
    for j in range(2,i):
        if i%j == 0:
            is_prime = False
            break
    if is_prime:
        prime_list.append(i)

print(prime_list)

# 斐波那契数列 前20个数
Fibonacci_list = []
a,b =1,1
for _ in range(20):
    Fibonacci_list.append(a)
    a,b = b,a+b
print(Fibonacci_list)

# 水仙花数 寻找100-999中所有水仙花数 如153 = 1³ + 5³ + 3³,N位非负整数，其各位数字的 N次方和刚好等于该数本身
Narcissistic_list=[]
for num in range(100,1000):
    i = num//100 # 取百位
    j = num//10%10 # 153//10=15，15%10=5 取十位
    k = num%10 # 153%10=3 取个位
    if i**3 + j**3 + k**3 == num:
        Narcissistic_list.append(num)
print(Narcissistic_list)

# 百钱百鸡问题 公鸡 5 元一只，母鸡 3 元一只，小鸡 1 元三只，用 100 块钱买一百只鸡，问公鸡、母鸡、小鸡各有多少只？
# 5x + 3y + (z/3) = 100/z为3的倍数/公鸡0-20只，母鸡0-33只，小鸡0-100只
for i in range (21):
    for j in range(34):
        k = 100 - i - j
        if k>=0 and k%3==0 and 5*i+3*j+k//3==100:
            print(f'公鸡{i}只，母鸡{j}只，小鸡{k}只')

# Day 08 列表

items6 = [45, 58, 29]
items7 = ['Python', 'Java', 'JavaScript']
print(items6 + items7)  # [45, 58, 29, 'Python', 'Java', 'JavaScript']
print(99 in items6)  # False
print('C++' not in items7)     # True
print(items7[0])   # Python
print(items7[-1])   # JavaScript