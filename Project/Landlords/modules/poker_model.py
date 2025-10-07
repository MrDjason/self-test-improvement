# ==================== 导入模块 ====================
import random
# ==================== 定 义 类 ====================
class Suite():
    '''花色'''
    SPADE, HEART, CLUB, DIAMOND, JOKER = range(5) # 定义类属性，方便直接调用

class Card():
    '''牌'''
    def __init__(self, suite, face):
        self.suite = suite
        self.face = face

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
    
    @property
    def has_next(self):
        return self.current < len(self.cards)
   
