# Day 20 面对对象编程应用
# ==================== 导入模块 ====================
from enum import Enum
import random
# 类和类之间的关系可以粗略的分为 is-a关系（继承）、**has-a关系（关联）**和 use-a关系（依赖）

# ==================== 定义 类 ====================
class Suite(Enum):
    """花色(枚举类)"""
    SPADE, HEART, CLUB, DIAMOND = range(4)
    # 枚举的本质就是：给 “固定的几个选项” 起个 “一看就懂的名字”，并且强制只能用这几个选项，不能乱改。
    # 这样表示花色就可以不用数字，而是用Suite.SPADE等

"""
class NormalClass:
    a = 1
    b = 2
这里的 a 和 b 是「类属性」，可以通过 NormalClass.a 直接访问，得到的是 1 这个值
Enum 枚举类不一样
class Suite(Enum):
    SPADE, HEART, CLUB, DIAMOND = range(4)
Python 会为每一个名称创建一个枚举类对象，给对象赋予两个值——name 和value
print(Suite.SPADE)       # Suite.SPADE 说明是Suite类的一个成员
print(type(Suite.SPADE)) # <enum 'Suite'> 说明是Suite的实例
print(Suite.SPADE.name)  # SPADE
print(Suite.SPADE.value) # 0
Suite.SPADE 是一个带有name和value枚举属性的对象
"""
for suite in Suite:
    print(f'{suite}:{suite.value}')

class Card:

    def __init__(self, suite, face): # 这里的 suite 参数，必须传入 Suite 枚举的成员，保证花色有效性
        self.suite = suite
        self.face = face

    def __repr__(self):
        suites = '♠♥♣♦'
        faces = ['', 'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        # ''的目的是为了让Pocker类里面的for face in range(1,14)能从1到13取值对应牌面
        return f'{suites[self.suite.value]}{faces[self.face]}'
        # self.suite 是卡牌花色属性，它的值必须是Suite 枚举类的某个成员
        # self.suite.value 会得到枚举的值(0-3),suites[self.suite.value]即用对象本身通过枚举获得的值对应花色

card1 = Card(Suite.SPADE,5) # Suite.SPADE 会被赋值给 self.suite
card2 = Card(Suite.SPADE,13)
print(card1)
print(card2)

class Poker:
    # 定义扑克类
    def __init__(self):
        self.cards = [Card(suite, face)
                      for suite in Suite
                      for face in range(1,14)] # 生成52个类的列表
        self.current = 0 # 记录发牌位置属性 表示[从第0张牌开始发牌，列表索引从0开始]

    def shuffle(self):
        # 洗牌
        self.current = 0           # 重置发牌位置到开头
        random.shuffle(self.cards) # 打乱牌的顺序
        # random.shuffle(self.cards)会直接修改self.cards列表，将其中的牌随机排序

    def deal(self):
        # 发牌
        card = self.cards[self.current] # 取出当前位置的牌
        self.current +=1                # 发牌位置后移一位
        return card                     # 返回这张牌
    
    @property
    # 装饰器，让这个方法可以像属性一样访问
    def has_next(self):
        # 还有无牌可发
        return self.current < len(self.cards)
        # 如果当前发牌位置 < 总牌数(52)，说明还有牌 
    
poker = Poker() # 创建一个Pocker类的实例对象
print(poker.cards) # 洗牌前的牌   调用Pocker的实例属性并打印，存储一个52个Card对象的列表
# 打印时Python会自动对每个Card对象调用其 __repr__ 方法
poker.shuffle() # 调用实例的洗牌方法，打乱实例属性中52个Card对象的顺序
print(poker.cards)  # 洗牌后的牌

class Player:
    '''玩家'''
    def __init__(self, name):
        self.name = name
        self.cards = [] # 玩家手上的牌
    
    def get_one(self, card):
        '''摸牌'''
        self.cards.append(card)
    
    def arrange(self):
        '''整理'''
        self.cards.sort()

# 创建四个玩家并发牌到玩家手上
poker = Poker()
poker.shuffle()
players = [Player('东'), Player('南'), Player('西'), Player('北')]
for _ in range(13):
    for player in players:
        player.get_one(poker.deal())
# 玩家整理手上的牌输出名字和手牌
for player in players:
    try:
        player.arrange()
        print(f'{player.name}: ', end='')
        print(player.cards)
    except TypeError as e:
        print(f'错误原因:{e}')
        print('arrange方法使用列表的sort进行排序比较两个Card对象的大小，而<运算符不能直接用于Card类型')

class Card:
    """牌"""

    def __init__(self, suite, face):
        self.suite = suite
        self.face = face

    def __repr__(self):
        suites = '♠♥♣♦'
        faces = ['', 'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        return f'{suites[self.suite.value]}{faces[self.face]}'
    
    def __lt__(self, other):
        if self.suite == other.suite:
            return self.face < other.face # 花色相同比较点数大小
        return self.suite.value < other.suite.value # 花色不同比较花色对应的值
    
from abc import ABCMeta, abstractmethod

# 工资结算系统

class Employee(metaclass=ABCMeta): # metaclass=ABCMeta：让Employee 成为抽象基类（ABC），不能直接创建实例
    '''员工'''
    def __init__(self, name):
        self.name = name 
    
    @abstractmethod
    # 抽象方法装饰器：只有声明、没有实现的方法,强制子类必须重写这个方法
    def get_salary (self):
        '''结算月薪'''
        pass

class Manager(Employee): # 继承父类Employee
    '''部门经理'''
    def get_salary(self): # 重写父类的抽象方法 get_salary（必须重写，否则报错）
        return 15000.0
# Manager没有额外属性（只有姓名，父类已处理），所以不用重写 __init__ 方法


class Programmer(Employee):
    '''程序员'''
    def __init__(self, name, working_hour =0): # 重写 __init__：程序员有额外属性“工作时长”
        super().__init__(name) # 调用父类 __init__ 方法：让父类初始化“姓名”属性（复用父类逻辑）
        self.working_hour = working_hour

    def get_salary(self):
        return 200 * self.working_hour # 程序员时薪200元
    
class Salesman(Employee):
    '''销售员'''
    def __init__(self, name, sales=0):
        super().__init__(name) # 调用父类 __init__方法，让父类初始化姓名属性
        self.sales = sales 

    def get_salary(self):
        return 1800 + self.sales * 0.05 
    
emps = [Manager('刘备'), Programmer('诸葛亮'), Manager('曹操'), Programmer('荀彧'), Salesman('张辽')]
for emp in emps:
    if isinstance(emp, Programmer):
    # isinstance(emp, 类型)：判断当前员工 emp 属于哪种类型
    # （Python 内置函数，用于检查对象是否是某个类的实例）
        emp.working_hour = int(input(f'请输入{emp.name}本月工作时间: '))
    elif isinstance(emp, Salesman):
        emp.sales = float(input(f'请输入{emp.name}本月销售额: '))
    print(f'{emp.name}本月工资为: ￥{emp.get_salary():.2f}元')
# 利用了面向对象的多态特性：不管是哪种员工，都可以用 emp.get_salary() 调用其对应的薪资计算逻辑