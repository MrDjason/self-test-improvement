# ==================== 导入模块 ====================
from enum import Enum
import random
# ==================== 定 义 类 ====================
class Suite(Enum):
    SPADE,HEART,CLUB,DIAMOND,JOKER=range(5)

class Card():
    def __init__(self, suite, face):
        self.suite = suite
        self.face = face

    def __lt__(self, other):
        # __lt__ 是 Python 中用于定义 “<” 比较规则的魔术方法
        # 1.让自定义类的实例支持 < 运算符(比较逻辑为 self < other, 成立则 return True)
        # 2.使实例列表能通过 sort() 或 sorted() 排序(sort 排序列表, sorted 排序所有可迭代对象)

        # 先判断王牌情况
        if self.suite == Suite.JOKER:  # self 是王牌情况
            if self.face == 1:         # 如果为大王牌 
                return False           # self < other 不可能成立
            else:                      # 判断 other 为王牌情况
                return other.suite == Suite.JOKER and other.face == 1
        if other.suite == Suite.JOKER: # self 不是王牌, other 是王牌
            return True                # 普通牌 < 王牌

        # 普通牌比较
        # 定义牌面大小、花色顺序
        faces = {2: 15, 1: 14, 13: 13, 12: 12, 11: 11, 10: 10, 9:9, 8:8, 7:7, 6:6, 5:5, 4:4, 3:3}      
        suites = {Suite.SPADE: 4, Suite.HEART: 3, Suite.CLUB: 2, Suite.DIAMOND: 1}

        if faces[self.face] != faces[other.face]:
            return faces[self.face] < faces[other.face]     # 两者面值不符则判断面值大小
        else:
            return suites[self.suite] < suites[other.suite] # 两者面值相等则判断花色顺序 

    def __repr__(self):
        suites = '♠♥♣♦🃏'
        faces = ['', 'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        if self.suite == Suite.JOKER:
            if self.face == 0:
                return '🃏小王'
            else:
                return '🃏大王'
        else:
            return f'{suites[self.suite.value]}{faces[self.face]}'
    
class Poker():
    def __init__(self):
        self.cards = [Card(suite, face) 
                 for suite in [Suite.SPADE, Suite.HEART, Suite.CLUB, Suite.DIAMOND]
                 for face in range(1,14)]
        self.cards.append(Card(Suite.JOKER, 0))
        self.cards.append(Card(Suite.JOKER, 1))
        self.current = 0

    def deal(self):
        card = self.cards[self.current]
        self.current += 1
        return card

    def shuffle(self):
        self.current = 0
        random.shuffle(self.cards)

    @property # @property 装饰器:把一个方法转换为只读属性
    # 调用效果:poker.has_next() -> poker.has_next
    def has_next(self):
        return self.current < len(self.cards)
