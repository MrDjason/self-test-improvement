# Day 30 正则表达式
# ==================== 导入模块 ====================
import re
# ==================== 主 程 序 ====================
username = input('请输入用户名：')
qq=input('请输入QQ号：')
m1 = re.match(r'^[0-9a-zA-Z_]{6,20}$', username)
if not m1: # 字符串不符合正则规则
    print('请输入有效用户名')

m2 = re.fullmatch(r'[1-9]\d{4,11}', qq)
if not m2:
    print('请输入有效QQ号')

if m1 and m2:
    print('你输入的信息是有效的！')

pattern = re.compile(r'(?<=\D)1[34578]\d{9}(?=\D)')
# (?<=\D)正向后顾断言,当前位置前必须为非数字字符
# 1[34578]\d{9} 号码为1+3或4或5或7或8+9位数字
# (?=\D) 正向前瞻断言，当前位置后面必须是一个非数字字符


sentence = '''重要的事情说8130123456789遍，我的手机号是13512346789这个靓号，
不是15600998765，也不是110或119，王大锤的手机号才是15600998765。'''

# 查找所有匹配并保存到一个列表中
tels_list = re.findall(pattern, sentence) # 
for tel in tels_list:
    print(tel)

# 通过迭代器取出匹配对象并获得匹配的内容
for temp in pattern.finditer(sentence): # finditer 找出所有非重叠的匹配项，但返回一个迭代器
    print(temp.group())

# 通过search函数指定搜索位置找出所有匹配
m = pattern.search(sentence) # search 从指定位置(pos)向后查找第一个匹配项，找到则返回匹配对象
while m:
    print(m.group())
    m = pattern.search(sentence, m.end())


# 替换
sentence = 'Oh, shit! 你是傻逼吗? Fuck you.'

purified = re.sub('fuck|shit|[傻煞沙][比笔逼叉缺吊碉雕]',
                  '*', sentence, flags=re.IGNORECASE)
print(purified)  # Oh, *! 你是*吗? * you.

# 拆分长句子
poem = '窗前明月光，疑是地上霜。举头望明月，低头思故乡。'
sentences_list = re.split(r'[，。]', poem)
sentences_list = [sentence for sentence in sentences_list if sentence]
for sentence in sentences_list:
    print(sentence)