# Day 08 列表

items6 = [45, 58, 29]
items7 = ['Python', 'Java', 'JavaScript']
print(items6 + items7)  # [45, 58, 29, 'Python', 'Java', 'JavaScript']
print(99 in items6)  # False
print('C++' not in items7)     # True
print(items7[0])   # Python
print(items7[-1])   # JavaScript

# 切片
items = ['a', 'b', 'c', 'd', 'e', 'f']
print(items[0:4:2]) # ['a', 'c']
print(items[::2])   # ['a', 'c', 'e']

print(items[:])  # 输出：['a', 'b', 'c', 'd', 'e', 'f']（与原列表完全相同）
print(items[2::])    # 从索引2到末尾：['c', 'd', 'e', 'f']
print(items[:3:])    # 从开头到索引2：['a', 'b', 'c']
print(items[::2])   # 步长2（正向）：['a', 'c', 'e']

items[1:1] = ['BB', 'CC'] # [1，1)没有空间，items[1]后插入数据 BB、CC
print(items)
items[1:2] = ['0', '0'] # [1,2）有空间，items[1]修改 BB→0,items[1]后插入0
print(items)
# 遍历
languages = ['l', 'a', 'n', 'g', 'u', 'a', 'g', 'e', 's']
for language in languages:
    print(language,end = ' ')

for index in range(len(languages)):
    print(languages[index], end = ' ')


