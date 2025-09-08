# 添加元素
languages = ['Java', 'C', 'C++', 'Python']
languages.append('JavaScript')
print(languages)
languages.insert(1,'SQL') # ['Java', 'SQL', 'C', 'C++', 'Python', 'JavaScript']
print(languages)

# 删除元素
languages.pop() # 无参数默认删除最后一位 ['Java', 'SQL', 'C', 'C++', 'Python']
temp = languages.pop(1)
print(temp) # SQL
languages.clear()
print(languages) # []

languages = ['Python', 'Python', 'C', 'C++']
languages.remove('Python') # 删除第一个Python
print(languages)

del languages[1] # 只删除Python，而pop可用temp进行存储

# 查找元素位置
items = ['Python', 'Java', 'Java', 'C++', 'Kotlin', 'Python']
print(items.index('Python')) # 查找元素位置 0
print(items.index('Java', 1)) # 从1开始查找Java 1 
print(items.count('Python')) # 2

# 元素的排序和倒转
items = ['Python', 'Java', 'C++', 'Kotlin', 'Swift']
items.sort()
print(items) # sort从小到大进行排序如123，abc
items.reverse()
print(items) # 倒转

# 列表生成式
# 例一
items = []
for i in range(1,101):
    if i % 3 == 0 or i % 5 ==0:
        items.append(i)
print(items)

num_list = [i for i in range(1,101) if i % 3 == 0 or i % 5 == 0]
print(num_list)

# 例二
num_list_1 = [35, 12, 97, 64, 55]
num_list_2 = []
for num in num_list_1:
    num_list_2.append(num**2)
print(num_list_2)

num_list_3 = [num**2 for num in num_list_1]
print(num_list_3)

# 例三
num_list_4 = []
for num in num_list_1:
    if num >50:
        num_list_4.append(num)
print(num_list_4)

num_list_5 = [num for num in num_list_1 if num > 50]
print(num_list_5)

# 嵌套列表
scores = [[100,99,98], [61,60,59], [0,1,2]]
print(scores[2][1])

# 嵌套列表 列表生成式
import random

scores = []
for _ in range(5):
    temp = []
    for _ in range(3):
        temp.append(random.randrange(60, 101))
    scores.append(temp)
print(scores)

scores = [[random.randrange(60,101) for _ in range(3)] for _ in range(5)]
print(scores)