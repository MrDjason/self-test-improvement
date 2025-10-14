# ==================== 导入模块 ====================
# 无
# ==================== 定 义 类 ====================
class Player():
    def __init__(self, name):
    # 初始化玩家    
        self.name = name
        self.cards=[]

    def get_card(self, card):
    # 摸牌
        self.cards.append(card)

    def arrange(self):
    # 整理
        self.cards.sort(reversed=True)
