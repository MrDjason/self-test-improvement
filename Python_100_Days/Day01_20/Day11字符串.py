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
s1 = '\it \is \time \to \read \now'
s2 = r'\it \is \time \to \read \now'
print(s1)
print(s2)

# 字符的特殊表示