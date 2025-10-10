 class Player():
    '''玩家'''
    def __init__(self, name, point=0):
        '''初始化玩家'''
        self.name = name
        self.cards = []
    
    def get_card(self, card):
        '''摸牌'''
        self.cards.append(card)
    
    def arrange(self):
        '''整理'''
        self.cards.sort()
