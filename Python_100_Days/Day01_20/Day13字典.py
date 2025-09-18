# Day 13 字典
# 字典是 Python 中一种键值对（key-value）集合，用于存储具有映射关系的数据
# 键（key）必须是不可变类型（如字符串、数字、元组），且唯一（重复键会被覆盖）
# 值（value）可以是任意数据类型

person = {'name': '李雷', 'age': 22, 'height': 172, 'weight': 70}
print(person)  # {'name': '李雷', 'age': 22, 'height': 172, 'weight': 70}
person = dict(name='王大锤', age=55, height=168, weight=60)
print(person)  # {'name': '王大锤', 'age': 55, 'height': 168, 'weight': 60}

# 可以通过Python内置函数zip压缩两个序列并创建字典
# zip可将多个可迭代对象按位置打包成元组，只能被遍历一次，再次遍历会为空。
items1 = dict(zip('ABCDE', '12345'))
print(items1)  # {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5'}

# 用字典生成式语法创建字典
items3 = {x: x ** 3 for x in range(1, 6)}
print(items3)  # {1: 1, 2: 8, 3: 27, 4: 64, 5: 125}

person = {'name': '王大锤', 'age': 55, 'height': 168, 'weight': 60}

print(len(person))    # 4，字典中键值对的数量

for key in person:            # 遍历字典的键
    print(key, end=' ')       # name age height weight
print()
for value in person.values(): # 遍历字典的值
    print(value, end=' ')     # 王大锤 55 168 60

print('name' in person) # True
print('Tel' in person)  # False

person['name'] = '李雷'
person['age'] = 22
print(person) # {'name': '李雷', 'age': 22, 'height': 168, 'weight': 60}

# 字典的方法
print(person.get('name'))    # 李雷，获取键对应的值
print(person.keys())        # dict_keys(['name', 'age', 'height', 'weight'])，获取所有键
print(person.values())      # dict_values(['李雷', 22, 168, 60])，获取所有值    
print(person.items())       # dict_items([('name', '李雷'), ('age', 22), ('height', 168), ('weight', 60)])，获取所有键值对
# dict_items([('name', '李雷'), ('age', 22), ('height', 168), ('weight', 60)]) 是字典的键视图对象
print(person.pop('weight')) # 60，删除指定键值对并返回值
for key, value in person.items(): # 遍历所有键值对
    print(f"{key}:{value}", end=' ') # name:李雷 age:22 height:168
print()
# 更新
person1 = {'name': '王大锤', 'age': 42, 'height': 172}
person2 = {'name': '李雷', 'age': 22, 'height': 168, 'weight': 60}
person1.update(person2) # 用person2更新person1，有相同键则覆盖
print(person1)          # {'name': '李雷', 'age': 22, 'height': 168, 'weight': 60}

# 删除
# pop(key)
# 作用：删除指定key对应的键值对
# 返回：被删除key的值
# 注意：键不存在则抛KeyError
person1 = {'name': '李雷', 'age': 22, 'height': 168, 'weight': 60}
print(person.pop('age'))  # 返回22，字典中移除'age'键值对

# popitem()
# 作用：删除最后插入的键值对（无序字典中为随机）
# 返回：被删除的(key, value)二元组
print(person.popitem()) # 返回('weight', '...')，字典中移除该键值对

print(person)           # {'name': '李雷', 'height': 168}
print(person.clear())   # None，清空字典

person['gender'] = '男'
print(person)

# del关键字
# 作用：删除指定key对应的键值对
# 注意：键不存在则抛KeyError
while True:
    try:
        del person['age']  # 'age'键已被删除，抛KeyError
    except KeyError:
        print("键不存在，删除失败！")
        break
    del person['gender']  # 直接删除'age'键值对
    print(person)
    break
