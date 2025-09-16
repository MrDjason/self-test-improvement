# Day 10 元组

# 元组是不可变类型
# 元组类型变量一旦定义，其中元素不能增删， 元素值不能修改
# 若修改元组中的元素，会产生TypeError错误
t1 = (35,12,98)
t2 = ('李雷',45,True)

# 查看类型
print(type(t2[0])) # <class 'str'>

# 查看数量
print(len(t2)) # 3 

# 成员运算
print(12 in t1) # True
print(36 in t1) # False
print(t1 + t2)  # (35, 12, 98, '李雷', 45, True)

# 比较运算
t3 = (12,6,9)
print(t3 == (12,6,9))        # True
print((1,2,4) > (1,2,3))     # True
print((1,2,'e') > (1,2,'a')) # True

# 元组按元素数量有二元组、多元组等，空元组为 ()
# 单元素元组需加逗号（如 ('hello',)）
# 否则括号仅表运算优先级而非元组。
a = ()
print(type(a)) # <class 'tuple'>
b = ('Hello')  
print(type(b)) # <class 'str'>
c = ('Hello',)
print(type(c)) # <class 'tuple'>

# 当多个值用逗号分隔并赋予一个变量，多个值会被打包成元组类型
a = 1,10,100
print(type(a)) # <class 'tuple'>
print(a)       # (1, 10, 100)

# 把一个元组赋值给多个变量，元组会解包多个值分别赋予对应变量
i,j,k = a
print(i,j,k) # 1 10 100

# 如果解包出来时，元素个数和变量个数对不上，会引发ValueError异常
# 错误信息为：too many values to unpack 或 not enough values to unpack

a = 1, 10, 100, 1000
# i, j, k = a             # ValueError: too many values to unpack (expected 3)
# i, j, k, l, m, n = a    # ValueError: not enough values to unpack (expected 6, got 4)

# 有一种利于解决 变量个数少于元素个数 的方法，就是使用 * 表达式
# * 表达式修饰的变量会变成一个储存0或多个的列表，并且解包中* 表达式只能出现一次
a = 1,10,100,1000
i,j,*k = a
print(i,j,k)     # 1 10 [100, 1000]
*i,j = a
print(i,j)       # [1, 10, 100] 1000
i,j,k,l,*m = a
print(i,j,k,l,m) # 1 10 100 1000 []

# 多种综合
a, b, *c = range(1, 10)  
print(a, b, c)          # 1 2 [3, 4, 5, 6, 7, 8, 9]
a, b, c = [1, 10, 100]
print(a, b, c)          # 1 10 100
a, *b, c = 'hello'
print(a, b, c)          # h ['e', 'l', 'l'] o

# 交换变量的值
a, b, c = 1,2,3
print(a, b, c) # 1 2 3
a, b, c = b, c, a
print(a, b, c) # 2 3 1

# 元组和列表的比较

# 为什么Python已经有了列表类型，还要引入元组？
# 元组不可变，更适合多线程环境，【降低并发访问变量的同步化开销】
# 元组不可变，创建更快，因元素数量固定可精确分配内存，无需为动态修改预留空间
import timeit
print('%.3f 秒' % timeit.timeit('[1,2,3,4,5,6,7,8,9]', number=10000000)) # 0.917 秒
print('%.3f 秒' % timeit.timeit('(1,2,3,4,5,6,7,8,9)', number=10000000)) # 0.138 秒

# 列表转换元组
t1 = ['李雷',45,True]
t2 = tuple(t1)
print(type(t2)) # <class 'tuple'>
t3 = list(t2)
print(type(t3)) # <class 'list'>