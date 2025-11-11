# Day 12 集合

# 集合：无序不重复元素的序列
# 用大括号{}或set()函数创建集合
# {}表示空字典，空集合必须用set()函数创建

set1 = {1,2,3,3,3,2}
print(set1) # {1, 2, 3}
set2 = {'banana', 'pitaya', 'apple', 'apple', 'banana', 'grape'}
print(set2) # {'banana', 'grape', 'apple', 'pitaya'}
set3 = set('Hello')
print(set3) # {'H', 'e', 'l', 'o'}
set4 = set([1,2,2,3,3,3,2,1])
print(set4) # {1, 2, 3}
set5 = {num for num in range(1,20) if num % 3 == 0 or num % 7 == 0}
print(set5) # {3, 6, 7, 9, 12, 14, 15, 18}

# 集合中的元素必须为 hashalbe 类型
# 通常不可变类型都是 hashable 类型，如 int, float, bool, str, tuple 等

# 元素的遍历
set1 = {'Python', 'Java', 'C', 'C++', 'Go', 'JavaScript'}
for elem in set1:        # 元素无序
    print(elem, end=' ') # Python Java C++ C Go JavaScript
print()                  # 换行

# 成员运算
set1 = {1,2,3,4,5}
print(10 in set1)  # False
print(3 in set1)   # True

# 二元运算
set1 = {1,2,3,4,5,6,7}
set2 = {2,4,6,8,10}
# 交集
print(set1 & set2)             # {2, 4, 6} 交集
print(set1.intersection(set2)) # {2, 4, 6} 交集
# 并集
print(set1 | set2)      # {1, 2, 3, 4, 5, 6, 7, 8, 10} 并集
print(set1.union(set2)) # {1, 2, 3, 4, 5, 6, 7, 8, 10} 并集
# 差集
print(set1 - set2)            # {1, 3, 5, 7} 差集
print(set1.difference(set2))  # {1, 3, 5, 7} 差集
print(set2 - set1)            # {8, 10} 差集
print(set1 ^ set2)                     # {1, 3, 5, 7, 8, 10} 对称差集
print(set1.symmetric_difference(set2)) # {1, 3, 5, 7, 8, 10} 对称差集

set1 = {1, 3, 5, 7}
set2 = {2, 4, 6}
set1 |= set2
# set1.update(set2)
print(set1)  # {1, 2, 3, 4, 5, 6, 7}
set3 = {3, 6, 9}
set1 &= set3
# set1.intersection_update(set3)
print(set1)  # {3, 6}
set2 -= set1
# set2.difference_update(set1)
print(set2)  # {2, 4}

# 比较运算
set1 = {1,3,5}
set2 = {1, 2, 3, 4, 5}
set3 = {5, 4, 3, 2, 1}
print(set2 > set1)  # True set2 包含 set1
print(set2 >= set1) # True set2 包含 set1
print(set2 == set3) # True set2 等于 set3,无关顺序

print(set1.issubset(set2))   # True set1 是 set2 的子集
print(set2.issuperset(set1)) # True set2 是 set1 的超集

# 集合的方法
set1 = {1 ,10 ,100}
# 增加元素
set1.add(1000)
set1.add(10000)
set1.add(100)    # 元素已存在，集合不变
print(set1)      # {1, 10, 100, 1000, 10000}
# 删除元素
set1.discard(10) # 元素不存在时，不会报错
if 100 in set1:
    set1.remove(100) # 元素不存在时，会报错
    try:
        set1.remove(100)
    except KeyError:
        print("元素不存在，删除失败！")
print(set1) # {1, 1000, 10000} 
# 随机删除元素
elem = set1.pop()
print("被删除的元素是：", elem) # 被删除的元素是：
# 清空元素
set1.clear()
print(set1) # set()

set1 = {'Java', 'Python', 'C', 'C++', 'Go'}
set2 = {'Swift', 'John', 'Java', 'Kitty', 'Ruby'}
set3 = {'HTML', 'CSS', 'JavaScript'}

print(set1.isdisjoint(set2)) # False set1 和 set2 有交集
print(set1.isdisjoint(set3)) # True set1 和 set3 无交集

# frozenset 不可变集合
"""
特点——
1)不可修改：无法添加、删除元素（无 add()/remove() 等方法）
2)可哈希：可作为 set 的元素（普通 set 不行）
3)其他特性：与 set 一致（无序、元素唯一）
"""
 
# fset = frozenset(可迭代对象)
fset1 = frozenset([1, 3, 5, 6])
fset2 = frozenset([2, 4, 6])

print(fset1 & fset2) # {6} 交集
print(fset1 | fset2) # {1, 2, 3, 4, 5, 6} 并集
print(fset1 - fset2) # {1, 3, 5} 差集
print(fset1 ^ fset2) # {1, 2, 3, 4, 5} 对称差集