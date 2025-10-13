# ==================== 导入模块 ====================
from modules.player_model import *
from modules.poker_model import *
from modules.Landlord_Rules import *

# ==================== 程    序 ====================
def landlords():
    """斗地主游戏主函数"""
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
    
    # 抢地主逻辑
    landlord_index = bid_landlord(players)
    landlord = players[landlord_index]
    print(f"{landlord.name} 成为地主！")
    
    # 地主获得底牌
    for card in bottom_cards:
        landlord.get_card(card)
    landlord.arrange()
    print(f"地主{landlord.name}的牌（包含底牌）：{landlord.cards[::-1]}")
    
    # 开始游戏
    play_game(players, landlord_index)


def bid_landlord(players):
    """抢地主逻辑"""
    print("\n开始抢地主！")
    landlord_bids = [-1] * len(players)  # -1:未叫, 0:不叫, 1:叫地主
    
    # 随机选择一个玩家开始叫地主
    current_player = 0
    while True:
        player = players[current_player]
        print(f"{player.name}，是否叫地主？(y/n): ")
        choice = input().strip().lower()
        
        if choice == 'y':
            landlord_bids[current_player] = 1
            # 如果有人叫地主，其他玩家可以选择抢地主（本简化版中直接结束）
            return current_player
        else:
            landlord_bids[current_player] = 0
            
        # 如果所有人都不叫地主，则重新开始
        if all(bid == 0 for bid in landlord_bids):
            print("所有人都不叫地主，重新开始抢地主！")
            landlord_bids = [-1] * len(players)
            
        current_player = (current_player + 1) % len(players)


def play_game(players, landlord_index):
    """游戏主循环"""
    print("\n游戏开始！")
    
    # 游戏状态
    current_player = landlord_index  # 地主先出牌
    last_player = -1  # 上一个出牌的玩家索引
    last_cards = []   # 上一手牌
    consecutive_passes = 0  # 连续跳过的玩家数量
    
    # 游戏主循环
    while True:
        player = players[current_player]
        print(f"\n{player.name}的回合")
        print(f"你的牌: {player.cards[::-1]}")
        
        # 检查是否所有人都跳过，重置状态
        if consecutive_passes == len(players):
            print("\n所有人都跳过，重新开始一轮！")
            last_cards = []
            last_player = -1
            consecutive_passes = 0
        
        # 如果是第一个出牌或者上一个出牌的是其他玩家
        if not last_cards or last_player != current_player:
            # 玩家需要出牌
            cards_input = input("请输入要出的牌的索引（例如: 0,1,2 或 pass跳过，但第一次出牌不能跳过）: ")
            
            # 第一次出牌不能跳过
            if cards_input.lower() == 'pass' and not last_cards:
                print("第一次出牌不能跳过！")
                continue
                
            # 跳过
            if cards_input.lower() == 'pass' and last_cards:
                print(f"{player.name} 选择跳过")
                consecutive_passes += 1
            else:
                # 处理玩家出牌
                try:
                    # 解析输入的索引
                    indices = [int(i.strip()) for i in cards_input.split(',')]
                    # 验证索引是否有效
                    if any(i < 0 or i >= len(player.cards) for i in indices):
                        print("无效的索引！")
                        continue
                    
                    # 获取要出的牌
                    selected_cards = [player.cards[i] for i in indices]
                    
                    # 判断牌型是否有效
                    if not LandlordRules.is_valid_play(selected_cards):
                        print("无效的牌型！")
                        continue
                    
                    # 判断是否能压过上一手牌
                    # 如果所有人都跳过了，则可以出任何有效牌型，不需要压过上一手牌
                    if last_cards and not (consecutive_passes == len(players) - 1 and current_player == last_player) and not LandlordRules.can_beat(last_cards, selected_cards):
                        print("不能压过上一手牌！")
                        continue
                    
                    # 出牌成功
                    last_cards = selected_cards
                    last_player = current_player
                    consecutive_passes = 0  # 重置连续跳过计数
                    
                    # 从玩家手中移除出的牌
                    for card in sorted(selected_cards, reverse=True):
                        player.cards.remove(card)
                    
                    print(f"{player.name} 出牌: {last_cards}")
                    
                    # 检查是否获胜
                    if not player.cards:
                        if current_player == landlord_index:
                            print(f"游戏结束！地主{player.name}获胜！")
                        else:
                            print(f"游戏结束！农民{player.name}获胜！")
                        return
                except ValueError:
                    print("输入格式错误！请输入逗号分隔的索引数字。")
                    continue
        
        # 如果有人出牌后，其他所有玩家都选择了pass，则由出牌的玩家继续出牌
        # 否则正常切换到下一个玩家
        if last_cards and consecutive_passes == len(players) - 1:
            # 所有其他玩家都pass了，让上一个出牌的玩家继续出牌
            current_player = last_player
            # 重置连续跳过计数，避免无限循环
            consecutive_passes = 0
        else:
            # 正常切换到下一个玩家
            current_player = (current_player + 1) % len(players)

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