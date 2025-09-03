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
