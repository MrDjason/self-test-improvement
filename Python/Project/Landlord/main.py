# ==================== 导入模块 ====================
from modules.player_model import *
from modules.poker_model import *
from modules.landlord_rule import *
# ==================== 游    戏 ====================
def landlords():
    while True:
    # 创建扑克
        poker = Poker()
        poker.shuffle()
        # 创建玩家
        players = [Player(f'玩家{i+1}') for i in range(3)]
        # 玩家发牌
        for i in range(17):
            for player in players:
                if poker.has_next:
                    card = poker.deal()
                    player.get_card(card)
        for player in players:
            player.arrange()
            print(f'{player.name}手牌:{player.cards}')
        # 留三张底牌
        bottom_cards = [poker.deal() for _ in range(3)]
        print(f'底牌:{bottom_cards}')
        # 抢地主
        current_player = landlord_bid(players)
        if current_player != -1:
            break
        else:
            print("所有人都不叫地主，将重新发牌...\n")
    landlord = players[current_player]
    print(f'{landlord.name}成为地主！')
    for card in bottom_cards:
        landlord.get_card(card)
    landlord.arrange()
    print(f'地主{landlord.name}的牌为:{landlord.cards}')
    play_game(players, current_player)

def play_game(players, landlord_index):
    '''斗地主游戏'''
    print('游戏开始！')
    current_player = landlord_index
    last_player = -1
    last_cards = []
    consecutive_passes = 0 # 连续跳过数
    is_new_round = False
    while True:
        current_idx = current_player
        player = players[current_idx]
        print(f"\n{player.name}的回合")
        print(f"你的牌: {player.cards}")

        # 判断是否出牌
        if (not last_cards) or is_new_round or last_player != current_player:
            is_new_round = False
            cards_input = input("请输入要出的牌的索引（例如: 0,1,2 或 pass跳过。第一次出牌不能跳过）: ")
            if cards_input.strip().lower() == 'pass' and not last_cards:
                print("第一次出牌不能跳过！")
                continue # 重新输入
            if cards_input.lower() == 'pass' and last_cards:
                print(f"{player.name} 选择跳过")
                consecutive_passes += 1
                if consecutive_passes == len(players)-1: # 如果其余两人都跳过
                    current_player = last_player  # 切回上一出牌者
                    consecutive_passes = 0
                    last_cards = []
                    is_new_round = True
                    print(f"\n所有玩家都跳过，{players[last_player].name}可自由出牌！")
                else:
                    current_player = (current_idx + 1) % len(players)  # 玩家2→玩家3
                # 跳过后续出牌逻辑，直接进入下一轮
                continue
                        # 3.3 处理“出牌”（校验+出牌）
            try:
                # 解析索引
                indices = [int(i.strip()) for i in cards_input.split(',')]
                # 校验索引：不重复、不越界
                if len(indices) != len(set(indices)): # 原切片与去重后不相等
                    print("无效输入！不能重复选择同一索引（如0,0）。")
                    continue  # 不切换，重新输入
                if any(i < 0 or i > len(player.cards) for i in indices): # 其中一个数字<0或者>玩家的所有牌则错误
                    print(f"无效索引！你的牌共{len(player.cards)}张，索引0~{len(player.cards)-1}。")
                    continue  # 不切换，重新输入

                # 选择并删除牌（倒序避免索引偏移）
                sorted_indices = sorted(indices, reverse=True)
                selected_cards = [player.cards[i] for i in sorted_indices]

                # 校验牌型和压制
                if not LandLordRules.is_valid(selected_cards):
                    print("无效的牌型！（需为单张、对子、顺子等合法牌型）")
                    continue  # 不切换，重新输入
                if last_cards and not LandLordRules.can_beat(last_cards, selected_cards):
                    print(f"不能压过上一手牌！上一手：{last_cards}")
                    continue  # 不切换，重新输入

                # 出牌成功：更新状态+删除牌
                last_cards = selected_cards
                last_player = current_idx
                consecutive_passes = 0
                for idx in sorted_indices:
                    del player.cards[idx]

                print(f"{player.name} 出牌: {last_cards}")

                # 检查获胜
                if not player.cards:
                    result = f"地主{player.name}" if current_idx == landlord_index else f"农民{player.name}"
                    print(f"\n===== 游戏结束！{result}获胜！=====")
                    return

                # 出牌成功→切换到下一个玩家
                current_player = (current_idx + 1) % len(players)

            except ValueError:
                print("输入格式错误！请用逗号分隔数字（如0,1,2）。")
                continue  # 不切换，重新输入

        # 4. 不需要出牌的情况（如自己刚出过牌，轮到他人）→ 直接切换
        else:
            current_player = (current_idx + 1) % len(players)
def landlord_bid(players: List):
    '''抢地主'''
    landlord_bid = [-1] * len(players)
    current_player = 0
    while True:
        player = players[current_player]
        print(f"{player.name}，是否叫地主？(y/n): ")
        choice = input().strip().lower()
        if choice == 'y':
            landlord_bid[current_player] = 1
            return current_player
        else:
            landlord_bid[current_player] = 0

            # 如果所有人都不叫地主，则重新开始
        if all(bid == 0 for bid in landlord_bid):
            print("所有人都不叫地主，重新开始抢地主！")
            return -1

        current_player = (current_player + 1) % len(players)
        # 玩家轮流操作的核心逻辑，作用是让玩家按顺序循环切换
        # 例如：有3个玩家，current_player初始为0，依次变为0,1,2,0,1,2...
        # (0+1)%3=1, (1+1)%3=2, (2+1)%3=0
# ==================== 主 界 面 ====================
def main():
    while True:
        print('='*20 + '欢迎来到斗地主游戏！' + '='*20)
        print("1. 开始游戏")
        print("2. 退出游戏")
        choice = input('请输入你的选择(1或者2):')
        if choice == '1':
            landlords()
        elif choice =='2':
            print('感谢使用，再见！')
            break
        else:
            print("无效输入，请重新选择！")

# ==================== 运    行 ====================
if __name__ == "__main__":
    main()