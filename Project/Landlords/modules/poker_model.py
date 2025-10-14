# ==================== 导入模块 ====================
import random
from enum import Enum
# ==================== 定 义 类 ====================
class Suite(Enum):
    '''花色'''
    SPADE, HEART, CLUB, DIAMOND, JOKER = range(5) # 定义枚举类，表示花色就可以不用数字，而是用Suite.SPADE等

class Card():
    '''牌'''
    def __init__(self, suite, face):
        self.suite = suite
        self.face = face

    def __lt__(self, other): 
        '''
        __lt__ 是 Python 中用于定义 “小于” 比较规则的魔术方法，它的核心作用是：
        1.让自定义类的实例支持 < 运算符；
        2.使实例列表能通过 sort() 或 sorted() 排序。 sort排序列表，sorted排序所有可迭代对象
        '''
        # other 是和当前对象(self)做比较的另一个对象
        # 比较逻辑 self < other
        if self.suite == Suite.JOKER: # 处理王牌
            if self.face == 1:  # 当前牌为大王
                return False  # 大王比任何牌大
            else:  # 当前牌为小王
                return other.suite == Suite.JOKER and other.face == 1  # 单独判断 小王 < 大王 情况，假如other是大王，返回True
        if other.suite == Suite.JOKER: # self 为普通牌，other 是王牌
            return True  # 普通牌 < 任何王
        
        # 普通牌比较：先比点数，点数相同再比花色
        # 定义点数大小（斗地主中 2 > A(1) > K(13) > ...）
        face_order = {2: 15, 1: 14, 13: 13, 12: 12, 11: 11, 10: 10, 9:9, 8:8, 7:7, 6:6, 5:5, 4:4, 3:3}
        # 定义花色大小（例如 黑桃 > 红桃 > 梅花 > 方块）
        suite_order = {Suite.SPADE: 4, Suite.HEART: 3, Suite.CLUB: 2, Suite.DIAMOND: 1}
        
        # 比较点数
        if face_order[self.face] != face_order[other.face]:
            return face_order[self.face] < face_order[other.face]
        # 点数相同则比较花色
        else:
            return suite_order[self.suite] < suite_order[other.suite]
        
    def __repr__(self):
        suites = '♠♥♣♦🃏'
        if self.suite == Suite.JOKER:
            if self.face == 0:
                return '🃏小王'
            else:
                return '🃏大王'
        else:
            faces = ['', 'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
            return (f'{suites[self.suite.value]}{faces[self.face]}')
    
class Poker():
    '''扑克'''
    def __init__(self):
        self.cards = [Card(suite, face)
                      for suite in [Suite.SPADE, Suite.HEART, Suite.CLUB, Suite.DIAMOND]
                      for face in range(1,14)]
        self.cards.append(Card(Suite.JOKER, 0))
        self.cards.append(Card(Suite.JOKER, 1))
        self.current = 0 # 记录发牌位置属性
    
    def shuffle(self):
        '''洗牌'''
        self.current = 0 # 重置发牌位置到开头
        random.shuffle(self.cards)
    
    def deal(self):
        '''发牌'''
        card = self.cards[self.current]
        self.current += 1
        return card
    
    @property # @property 装饰器:把一个方法转换为只读属性
    # 调用方法从:poker.has_next() -> poker.has_next
    def has_next(self):
        return self.current < len(self.cards)
   
