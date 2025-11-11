# Day 11 字符串

# 字符串：由零个或多个字符组成的有限序列

# 用单引号、双引号包围的字符串
s1 = 'hello, world!'
s2 = "你好，世界！❤️"
s3 = '''hello,
wonderful
world!'''
print(s1)
print(s2)
print(s3)

# 转义字符
# \后面的字符不再是他原来的意思，如\n代表换行
s1 = '\'hello, world!\'' # 'hello, world!'
s2 = '\\hello, world!\\' # \hello, world!\
print(s1)
print(s2)

# 原始字符串 r'\' 没有转义字符
s = r'\it \is \time \to \read \now'
print(s)

# 字符的特殊表示
s1 = '\141\142\143\x61\x62\x63' # 八进制 abcabc
s2 = '\u9a86\u660a' # 十六进制 骆昊
print(s1)
print(s2) 

# 字符串的运算

# 拼接和重复
s1 = 'hello' + ', ' + 'world' + '!'
print(s1)        # hello, world!
s2 = '!!!'
print(s1 + s2*3) # hello, world!!!!!!!!
s1 *=2
print(s1)        # hello, world!hello, world!

# 比较
# Python中字符串的比较都是通过Unicode编码逐个字符比较
# 核心规则是“逐个字符对比，从左到右依次判断”
print(ord('我')) # 25105 获取字符的Unicode编码

# 成员运算
s1 = 'hello'
s2 = 'world'
print('wo' in s1)     # False
print('or' in s2)     # True

# 获取字符串长度
s = 'hello, world!'
print(len(s)) # 13

# 索引和切片
s = 'hello, world!'
print(s[0], s[7])   # h w
print(s[-1], s[-3]) # ! r
print(s[0:5])       # hello
print(s[2:12])      # llo, world
print(s[:7])        # hello,
print(s[7:])        # world!
print(s[::2])       # hlo ol!

# 遍历字符串
# 法一：
s = 'hello, world!'
for i in range(len(s)):
    print(s[i], end=' ') # h e l l o ,   w o r l d !    

# 法二：
for element in s:
    print(element, end=' ') # h e l l o ,   w o r l d !    

# 字符串的方法
s = 'hello, world!'
# 字符串首字母大写
print(s.capitalize())    # Hello, world!
# 字符串每个单词首字母大写
print(s.title())         # Hello, World!
# 字符串变大写
print(s.upper())         # HELLO,WORLD!
# 字符串变小写
print(s.lower())         # hello,world!

# 字符串查找
s = 'hello, world!'
# find()方法作用：在字符串中查找子字符串，返回 “第一个出现的起始位置”；如果找不到，返回 -1
# index()方法作用：在字符串中查找子字符串，返回 “第一个出现的起始位置”；如果找不到，会引发ValueError异常
print(s.find('or'))      # 8(第一个出现的起始位置)
print(s.find('your'))    # -1
print(s.index('or'))     # 8(第一个出现的起始位置)
try:
    print(s.index('your'))  # 抛出异常
except ValueError as e:
    print(e)               # substring not found
# print(s.index('your'))   # ValueError: substring not found

# 在指定索引后查找
print(s.find('o', 1, 4)) # -1 在从1-4索引中查找o，没找到

# 性质判断
# startswith()：检查字符串是否以指定子字符串开头
# endswith()：检查字符串是否以指定子字符串结尾
s = 'hello, world!'
print(s.startswith('He'))  # False
print(s.startswith('hel')) # True
print(s.endswith('!'))     # True

# isdigit()：检查字符串是否只包含数字字符
# isalpha()：检查字符串是否只包含字母字符
# isalnum()：检查字符串是否只包含字母和数字字符
"""
num = 123
print(num.isdigit())        # True
错误！isdigit()是字符串的方法
"""
print('123'.isdigit())      # True
print('hello'.isalpha())    # True
print('hello123'.isalnum()) # True

# 格式化
# 字符串类型可通过center、ljust、rjust方法，做居中、左对齐、右对齐处理
# 左侧补零，可通过zfill方法
s = 'hello,world!'
print(s.center(20,'*'))  # ****hello,world!****
print(s.ljust(20, '-'))  # hello,world!-------
print(s.rjust(20, '+'))  # ++++++++hello,world!
print('33'.zfill(20))    # 00000000000000000033
print('-33'.zfill(5))    # -0033    

# 字符串格式化
a, b = 123,321
print('%d * %d = %d' % (a, b, a*b))     # 123 * 321 = 39483
print('{} * {} = {}'.format(a, b, a*b)) # 123 * 321 = 39483
print(f'{a} * {b} = {a*b}')              # 123 * 321 =39483

# 字符串修剪
s = '  hello, world!  '
print(s.strip())   # hello, world! 去掉两侧空白字符
s = '***hello, world!***'
print(s.lstrip('*')) # hello, world!*** 去掉左侧*
print(s.rstrip('*')) # ***hello, world! 去掉右侧*

# 替换操作
s = 'hello, world!'
print(s.replace('world', 'python')) # hello, python!
print(s.replace('o', 'e', 1))       # helle, world! 只替换第一个o

# 拆分与合并
s = 'I love you'
words = s.split('')    # ['I', 'love', 'you'] 以空格拆分字符串
print(words)
print('~'.join(words)) # I~love~you
words = s.split(' ',1) # ['I', 'love you'] 以空格拆分字符串，最多拆分1次

# 编码与解码
a = '你好世界'
b = a.encode('utf-8')
c = a.encode('gbk')
print(b)                 # b'\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c'
print(c)                 # b'\xc4\xe3\xba\xc3\xca\xc0\xbd\xe7'
print(b.decode('utf-8')) # 你好世界
print(c.decode('gbk'))   # 你好世界
try:
    print(b.decode('gbk')) # 出错
except UnicodeDecodeError as e:
    print(e)               # 'utf-8' codec can't decode byte 0xe4 in position 0: invalid continuation byte