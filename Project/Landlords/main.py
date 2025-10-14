# ==================== 导入模块 ====================
from modules.player_model import *
from modules.poker_model import *
from modules.Landlord_Rules import *

# ==================== 程    序 ====================
def landlords():
    """斗地主游戏主函数"""
    while True:
        # 创建扑克
        poker = Poker()
        poker.shuffle()
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
            print(f'{player.name}:{player.cards[:]}')     # 打印理牌后的结果

        # 剩余3张作为底牌
        bottom_cards = [poker.deal() for _ in range(3)]
        print(f"底牌：{bottom_cards}")
        
        # 抢地主逻辑
        landlord_index = bid_landlord(players)
        # 如果成功选出地主，退出循环
        if landlord_index != -1:
            break
        else:
            print("所有人都不叫地主，将重新发牌...\n")
    
    landlord = players[landlord_index]
    print(f"{landlord.name} 成为地主！")
    
    # 地主获得底牌
    for card in bottom_cards:
        landlord.get_card(card)
    landlord.arrange()
    print(f"地主{landlord.name}的牌（包含底牌）：{landlord.cards[:]}")
    
    # 开始游戏
    play_game(players, landlord_index)


def bid_landlord(players):
    """抢地主逻辑"""
    print("\n开始抢地主！")
    landlord_bids = [-1] * len(players)  # -1:未叫, 0:不叫, 1:叫地主
    # [-1] * len(players): 将[-1]这个列表重复len(players)次，生成一个新列表

    # 开始叫地主
    current_player = 0
    while True:
        player = players[current_player]
        print(f"{player.name}，是否叫地主？(y/n): ")
        choice = input().strip().lower()
        
        if choice == 'y':
            landlord_bids[current_player] = 1
            # 抢地主功能等待加入
            return current_player
        else:
            landlord_bids[current_player] = 0
            
        # 如果所有人都不叫地主，则重新开始
        if all(bid == 0 for bid in landlord_bids):
            print("所有人都不叫地主，重新开始抢地主！")
            return -1
                
        current_player = (current_player + 1) % len(players)
        # 玩家轮流操作的核心逻辑，作用是让玩家按顺序循环切换
        # 例如：有3个玩家，current_player初始为0，依次变为0,1,2,0,1,2...
        # (0+1)%3=1, (1+1)%3=2, (2+1)%3=0
        
# ==================== 游    戏 ====================
def play_game(players, landlord_index):
    """游戏主循环（彻底修复跳过不切换问题）"""
    print("\n游戏开始！")
    
    current_player = landlord_index  # 当前玩家索引（0=玩家1，1=玩家2，2=玩家3）
    last_player = -1  # 上一出牌者索引
    last_cards = []   # 上一手牌
    consecutive_passes = 0  # 连续跳过计数
    player_count = len(players)  # 固定3人
    is_new_round = False  # 新一轮出牌（可自由出牌）

    while True:
        # 1. 锁定当前玩家索引和对象（避免后续修改）
        current_idx = current_player
        player = players[current_idx]
        print(f"\n{player.name}的回合")
        print(f"你的牌: {player.cards[:]}")

        # 2. 处理“所有人全跳过”的重置
        if consecutive_passes == player_count:
            print("\n所有人都跳过，重新开始一轮！")
            last_cards = []
            last_player = -1
            consecutive_passes = 0
            is_new_round = True
            # 重置后：当前玩家继续出牌（不切换）
            continue  # 直接进入下一轮，当前玩家重新输入

        # 3. 判断是否需要出牌（只有3种情况需要出牌）
        need_play = (not last_cards) or (last_player != current_idx) or is_new_round
        if need_play:
            is_new_round = False  # 重置新一轮标志
            cards_input = input("请输入要出的牌的索引（例如: 0,1,2 或 pass跳过，但第一次出牌不能跳过）: ")

            # 3.1 第一次出牌（无last_cards）不能跳过
            if cards_input.lower() == 'pass' and not last_cards:
                print("第一次出牌不能跳过！")
                # 不切换玩家，重新输入
                continue

            # 3.2 处理“跳过”（核心修复：直接更新current_player，不依赖continue）
            if cards_input.lower() == 'pass' and last_cards:
                print(f"{player.name} 选择跳过")
                consecutive_passes += 1

                # 情况A：所有其他玩家都跳过（3人→跳过数=2）→ 回退到上一出牌者
                if consecutive_passes == player_count - 1:
                    current_player = last_player  # 切回上一出牌者（如玩家1）
                    consecutive_passes = 0
                    last_cards = []
                    is_new_round = True
                    print(f"\n所有玩家都跳过，{players[last_player].name}可自由出牌！")
                # 情况B：未全跳过→切换到下一个玩家（直接计算并更新）
                else:
                    current_player = (current_idx + 1) % player_count  # 玩家2→玩家3
                # 跳过后续出牌逻辑，直接进入下一轮
                continue

            # 3.3 处理“出牌”（校验+出牌）
            try:
                # 解析索引
                indices = [int(i.strip()) for i in cards_input.split(',')]
                # 校验索引：不重复、不越界
                if len(indices) != len(set(indices)):
                    print("无效输入！不能重复选择同一索引（如0,0）。")
                    continue  # 不切换，重新输入
                if any(i < 0 or i >= len(player.cards) for i in indices):
                    print(f"无效索引！你的牌共{len(player.cards)}张，索引0~{len(player.cards)-1}。")
                    continue  # 不切换，重新输入

                # 选择并删除牌（倒序避免索引偏移）
                sorted_indices = sorted(indices, reverse=True)
                selected_cards = [player.cards[i] for i in sorted_indices]

                # 校验牌型和压制
                if not LandlordRules.is_valid_play(selected_cards):
                    print("无效的牌型！（需为单张、对子、顺子等合法牌型）")
                    continue  # 不切换，重新输入
                if last_cards and not LandlordRules.can_beat(last_cards, selected_cards):
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
                current_player = (current_idx + 1) % player_count

            except ValueError:
                print("输入格式错误！请用逗号分隔数字（如0,1,2）。")
                continue  # 不切换，重新输入

        # 4. 不需要出牌的情况（如自己刚出过牌，轮到他人）→ 直接切换
        else:
            current_player = (current_idx + 1) % player_count

# ==================== 主 程 序 ====================
def main():
    print('=' * 20 +'欢迎来到斗地主游戏！'+ '=' * 20)
    while True:  # 保持主循环一直运行，直到用户选择退出
        print('1.开始游戏')
        print('2.退出游戏')
        command = input('请输入命令（1/2）：')
        if command == '1':
            # 开始游戏后，不使用break退出循环，游戏结束后会自动回到这个菜单
            landlords()
            # 游戏结束后打印分隔线，提示回到首页
            print('\n' + '=' * 50)
            print('游戏已结束，返回首页')
            print('=' * 50 + '\n')
        elif command == '2':
            print('退出游戏，感谢您的参与！')
            break  # 只有选择退出时才打破循环
        else:
            print('无效命令，请重新输入。')

# ==================== 运    行 ====================
if __name__ == '__main__':
    main()