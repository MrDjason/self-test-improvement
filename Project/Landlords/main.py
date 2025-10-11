# ==================== 导入模块 ====================
from modules.player_model import *
from modules.poker_model import *
from modules.rule_model import *

# ==================== 程    序 ====================
def landlords():
    # 创建扑克
    poker = Poker()
    print("洗牌前的牌序（前5张）：", poker.cards[:])  # 打印前5张看效果
    poker.shuffle()
    print("洗牌后的牌序（前5张）：", poker.cards[:])  # 洗牌后前5张变化
    # 创建玩家
    players = [Player(f'玩家 {i}') for i in range(1,4)]

    # 玩家发牌、理牌
    for i in range(17):
        for player in players:
            if poker.has_next:
                card = poker.deal()
                player.get_card(card)

        
    for player in players:
        player.arrange()
        print(f'{player.name}:{player.cards[::-1]}')     # 打印理牌后的结果

    # 剩余3张作为底牌
    bottom_cards = [poker.deal() for _ in range(3)]
    print(f"底牌：{bottom_cards}")

# ==================== 主 程 序 ====================
def main():
    print('=' * 20 +'欢迎来到斗地主游戏！'+ '=' * 20)
    while True:
        print('1.开始游戏')
        print('2.退出游戏')
        command = input('请输入命令（1/2）：')
        if command == '1':
            landlords()
            break
        elif command == '2':
            print('退出游戏，感谢您的参与！')
            break
        else:
            print('无效命令，请重新输入。')
    # 这里可以添加更多的游戏逻辑
# ==================== 运    行 ====================
if __name__ == '__main__':
    main()