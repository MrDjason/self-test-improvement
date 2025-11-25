# Day 21 文件读写和异常处理
# ==================== 基本信息 ====================
'''
'r'	读取 （默认）
'w'	写入（会先截断之前的内容）
'x'	写入，如果文件已经存在会产生异常
'a'	追加，将内容写入到已有文件的末尾
'b'	二进制模式
't'	文本模式（默认）
'+'	更新（既可以读又可以写）
'''
'''

'''
# ==================== 导入模块 ====================
import os
# ==================== 定 义 类 ====================
# ==================== 定义函数 ====================
# ==================== 主 程 序 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'res', '三国志.txt')
file_path_1 = os.path.join(script_dir, 'res', 'test1.jpg')
file_path_2 = os.path.join(script_dir, 'res', 'test2.jpg')
file = open(file_path, 'r', encoding = 'utf-8')
'''print(file.read())
file.close()
# 操作系统对同时打开的文件数量有上限，且文件打开后会占用内存、I/O 缓冲区等资源
# file.close()作用在于：1.释放系统资源2.刷新缓存数据3.确保文件完整性
file = open(file_path, 'r', encoding = 'utf-8')
for line in file:
    print(line)
file.close()'''

file = open(file_path, 'a', encoding = 'utf-8')
# 'a'为追加，w为截断之前文本内容并写入新内容
file.write('\n标题：《三国志》')
file.write('\n作者：陈寿')
file.write('\n时间：西晋')
file = open(file_path, 'r', encoding = 'utf-8')
for line in file:
    print(line)
file.close()

file = open(file_path, 'w', encoding = 'utf-8')
file.write('《三国志》是由西晋史学家陈寿耗时十年，参考当时的史书，编纂而成。'
'记载中国三国时期的曹魏、蜀汉、东吴的纪传体断代史，是二十四史中评价最高的“前四史”之一。\n第二段\n第三段')
file = open(file_path, 'r', encoding = 'utf-8')
print(file.read())
file.close()


'''
几大错误类型：
|BaseException        # 所有异常类的"根类"
+---SystemExit        # 程序主动结束的信号
+---KeyboardInterrupt # 用户强制中断程序时引发的异常
+---GeneratorExit     # 生成器（generator）被关闭时引发的异常
+---Exception         # 所有"常规异常"的基类
'''

class InputError(ValueError):
    '''自定义异常类型'''
    pass

def fac(num):
    '''求阶乘'''
    if num < 0:
        raise InputError('只能计算非负整数的阶乘')
    if num in (0, 1):
        return 1
    return num * fac(num-1)

flag = True
while flag:
    num = int(input('n = '))
    try:
        print(f'{num}! = {fac(num)}')
        flag = False
    except InputError as err:
        print(err)

# 上下文管理器语法
# 可以使用with上下文管理器语法在文件操作完成后自动执行文件对象的close方法
with open(file_path, 'r', encoding = 'utf-8') as file:
    print(file.read())

# 读写二进制文件

# 复制文件操作
try:
    with open(file_path_1, 'rb') as file1:
        # 读操作，使用'rb'
        data = file1.read()
    with open(file_path_2, 'wb') as file2:
        # 写操作，使用'wb'
        file2.write(data)
except FileNotFoundError:
    print('指定文件无法打开')
except IOError:
    print('读写文件出现错误')
print('程序结束')

try: 
    with open('test3.jpg', 'rb') as file1,open('test4.jpg', 'wb')as file2:
        # 从file1读取512字节二进制数据，存到data里面
        data = file1.read(512)
        while data:
        # data不是空的就继续循环
            file2.write(data)
            # 把本次读到的512字节数据，写入新文件file2
            data = file1.read()
            # 继续读取下一部分数据，读取后续内容直到末尾
except FileNotFoundError:
    print('指定文件无法打开')
except IOError:
    print('读写文件出现错误')
print('程序执行结束')
'''
假设复制一个 1GB 的大图片test3给test4，用read()一次性读完会瞬间占用1GB
可能会导致崩溃，而read(512)一次只会调用512字节

'''