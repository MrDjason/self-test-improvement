# Day 31 语言进阶
# ==================== 导入模块 ====================
import heapq
import itertools
from collections import Counter
# ==================== 主 程 序 ====================
prices = {
    'AAPL': 191.88,
    'GOOG': 1186.96,
    'IBM': 149.24,
    'ORCL': 48.44,
    'ACN': 166.89,
    'FB': 208.09,
    'SYMC': 21.29
}
# 用股票价格大于100元的股票构造一个新的字典
prices2 = {key:value for key, value in prices.items() if value>100}
print(prices2)

# 嵌套列表的坑
names = ['关羽', '张飞', '赵云', '马超', '黄忠']
courses = ['语文', '数学', '英语']

# 错误案例：scores = [[None]*3]*5
# 创建了[None, None, None] * 5
# 但是外层的列表是第一个列表的应用，指向内存同一个列表
# 第一个列表的某个值修改了，其余列表的值都跟着修改
'''
scores = [[None]*len(courses) for _ in range(len(names))]
for row,name in enumerate(names):
    for col, course in enumerate(courses):
        scores[row][col] = float(input(f'请输入{name}的{course}成绩：'))
        print(scores)
'''
# heapq模块
list1 = [34, 25, 12, 99, 87, 63, 58, 78, 88, 92]
list2 = [
    {'name': 'IBM', 'shares': 100, 'price': 91.1},
    {'name': 'AAPL', 'shares': 50, 'price': 543.22},
    {'name': 'FB', 'shares': 200, 'price': 21.09},
    {'name': 'HPQ', 'shares': 35, 'price': 31.75},
    {'name': 'YHOO', 'shares': 45, 'price': 16.35},
    {'name': 'ACME', 'shares': 75, 'price': 115.65}
]

print(heapq.nlargest(3, list1)) 
print(heapq.nsmallest(3, list1))
print(heapq.nlargest(2, list2, key=lambda x: x['price']))
# heapq.nlargest(n, iterable, key)
# lambda 参数:返回值
# 当heapq.nlargest处理list2时，会逐个取出列表中的字典
# lambda x:...就代表被取出来的这个字典
# 如取出list2的第一个元素，x = {'name': 'IBM', 'shares': 100, 'price': 91.1}
# 返回x[prices]对应的值
print(heapq.nlargest(2, list2, key = lambda x: x['shares']))

# itertools模块

# 产生ABCD的全排列 [('A', 'B', 'C', 'D'), ..., ('D', 'C', 'B', 'A')]
print(list(itertools.permutations('ABCD')))
# 产生ABCDE的五选三组合 [('A', 'B', 'C'), ..., ('C', 'D', 'E')]
print(list(itertools.combinations('ABCDE', 3)))
# 产生ABCD和123的笛卡尔积 [('A', '1'), ...,('D', '3')]
print(list(itertools.product('ABCD', '123')))
# 产生ABC的无限循环序列 
# print(list(itertools.cycle(('A', 'B', 'C'))))

# collections模块
words = [
    'look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes',
    'the', 'eyes', 'the', 'eyes', 'the', 'eyes', 'not', 'around',
    'the', 'eyes', "don't", 'look', 'around', 'the', 'eyes',
    'look', 'into', 'my', 'eyes', "you're", 'under'
]
counter = Counter(words)
print(counter.most_common(3)) # most_common(n) 返回出现次数最多的前 n 个元素

# 数据结构和算法