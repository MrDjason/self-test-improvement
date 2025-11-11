# Day 23 读写CSV文件
# ==================== 基本信息 ====================
'''
CSV文件介绍
CSV（Comma Separated Values）全称逗号分隔值文件是一种简单、通用的文件格式
被广泛的应用于应用程序（数据库、电子表格等）数据的导入和导出以及异构系统之间的数据交换

CSV文件有以下特点：

纯文本，使用某种字符集（如ASCII、Unicode、GB2312）等）；
由一条条的记录组成（典型的是每行一条记录）；
每条记录被分隔符（如逗号、分号、制表符等）分隔为字段（列）；

'''
# ==================== 导入模块 ====================
import csv
import os
import random
# ==================== 主 程 序 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'res', 'write_down.csv')

with open(file_path, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['姓名', '语文', '数学', '英语'])
    names = ['关羽', '张飞', '赵云', '马超', '黄忠']
    for name in names:
        scores = [random.randrange(50, 101) for _ in range(3)] # 例如：[85, 92, 78]
        scores.insert(0, name)  # 把姓名插入到列表第一个位置，变成 [姓名, 语文, 数学, 英语]
        writer.writerow(scores) # 此时 scores 示例：['关羽', 85, 92, 78]
        # writerow() 是这个对象的方法，功能是 将传入的可迭代对象（这里是 scores 列表）作为一行数据写入 CSV 文件
        # 列表中的每个元素会被自动作为 CSV 中的一列，元素之间会自动添加逗号

# dialect 表示CSV的方言，默认值是excel
# delimiter 分隔符
# quotechar 包围值的字符（默认“”）
# quoting 包围方式

with open(file_path, 'a') as file:
    writer = csv.writer(file, delimiter='|', quoting=csv.QUOTE_ALL) # csv.QUOTE_ALL强制给所有字段加双引号
    writer.writerow(['姓名', '语文', '数学', '英语'])
    names = ['关羽', '张飞', '赵云', '马超', '黄忠']
    for name in names:
        scores = [random.randrange(50, 101) for _ in range(3)] # 例如：[85, 92, 78]
        scores.insert(0, name)  # 把姓名插入到列表第一个位置，变成 [姓名, 语文, 数学, 英语]
        writer.writerow(scores)

# 从CSV文件读取数据 
with open(file_path, 'r') as file:
    reader = csv.reader(file, delimiter = '|')
    for data_list in reader: # reader会按行自动解析csv，每次循环会将所有字段赋值给data_list
        print(reader.line_num, end='\t')
        for elem in data_list:
            print(elem, end='\t')
        print()
# 两次写入，打印两次