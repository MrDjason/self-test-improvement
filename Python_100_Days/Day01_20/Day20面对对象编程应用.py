# Day 20 面对对象编程应用
from enum import Enum
# 类和类之间的关系可以粗略的分为 is-a关系（继承）、**has-a关系（关联）**和 use-a关系（依赖）

class Suite(Enum):
    """花色(枚举)"""
    SPADE, HEART, CLUB, DIAMOND = range(4)
    # 枚举的本质就是：给 “固定的几个选项” 起个 “一看就懂的名字”，并且强制只能用这几个选项，不能乱改。
    # 这样表示花色就可以不用数字，而是用Suite.SPADE等

for suite in Suite:
    print(f'{suite}:{suite.value}')

class Card:

    def __init__(self, suite, face):
        self.suite = suite
        self.face = face

    def repr(self):
        suites = '♠♥♣♦'
        faces = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        return f'{suites[self.suite.value]}{faces[self.face]}'
        # self.suite.value 会得到枚举的值(0-3),suites[self.suite.value]即用对象本身通过枚举获得的值对应花色

card1 = Card(Suite.SPADE,5)
card2 = Card(Suite.SPADE,13)
print(card1)
print(card2)

# 定义扑克类
class Poker: