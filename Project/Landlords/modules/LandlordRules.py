# ==================== 导入模块 ====================
import random
from enum import Enum
# ==================== 定 义 类 ====================
from enum import Enum
from typing import List, Tuple, Optional

class Suite(Enum):
    SPADE, HEART, CLUB, DIAMOND, JOKER = range(5)

class Card:
    def __init__(self, suite, face):
        self.suite = suite
        self.face = face
    
    def __lt__(self, other):
        if self.suite == Suite.JOKER:
            if self.face == 1:
                return False
            else:
                return other.suite == Suite.JOKER and other.face == 1
        if other.suite == Suite.JOKER:
            return True
        
        face_order = {2:15, 1:14, 13:13, 12:12, 11:11, 10:10, 9:9, 8:8, 7:7, 6:6, 5:5, 4:4, 3:3}
        suite_order = {Suite.SPADE:4, Suite.HEART:3, Suite.CLUB:2, Suite.DIAMOND:1}
        
        if face_order[self.face] != face_order[other.face]:
            return face_order[self.face] < face_order[other.face]
        else:
            return suite_order[self.suite] < suite_order[other.suite]
    
    def __repr__(self):
        suites = '♠♥♣♦🃏'
        if self.suite == Suite.JOKER:
            return '🃏大王' if self.face == 1 else '🃏小王'
        faces = ['', 'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        return f'{suites[self.suite.value]}{faces[self.face]}'


class LandlordRules:
    """斗地主出牌规则类（含6带2、6带4、4带2）"""
    
    class CardType(Enum):
        """更新后的牌型枚举"""
        INVALID = 0  # 无效牌型
        SINGLE = 1    # 单张
        PAIR = 2      # 对子
        TRIPLE = 3    # 三张
        STRAIGHT = 4  # 顺子(5+单张连续)
        STRAIGHT_PAIR = 5  # 连对(3+对子连续)
        TRIPLE_WITH_SINGLE = 6  # 三带一
        TRIPLE_WITH_PAIR = 7    # 三带二
        BOMB = 8      # 炸弹(4张相同)
        ROYAL_BOMB = 9  # 王炸(大小王)
        SIX_WITH_TWO = 10  # 6带2（2组3张+2张单牌或1对）
        SIX_WITH_FOUR = 11  # 6带4（2组3张+2对对子）
        FOUR_WITH_TWO = 12  # 4带2（炸弹+2张单牌或1对）
    
    @staticmethod
    def get_face_counts(cards: List[Card]) -> dict:
        """统计每张点数的出现次数（排除大小王）"""
        face_counts = {}
        for card in cards:
            if card.suite == Suite.JOKER:
                continue  # 这些牌型不允许包含大小王
            key = card.face
            face_counts[key] = face_counts.get(key, 0) + 1
        return face_counts
    
    @staticmethod
    def is_royal_bomb(cards: List[Card]) -> bool:
        """判断是否为王炸"""
        if len(cards) != 2:
            return False
        return (cards[0].suite == Suite.JOKER and cards[1].suite == Suite.JOKER and
                sorted([cards[0].face, cards[1].face]) == [0, 1])
    
    @staticmethod
    def get_card_type(cards: List[Card]) -> Tuple[CardType, int, Optional[Card]]:
        """判断牌型，返回(牌型, 关键值, 最大牌)"""
        if not cards:
            return (LandlordRules.CardType.INVALID, 0, None)
        
        sorted_cards = sorted(cards)
        max_card = sorted_cards[-1]
        card_count = len(sorted_cards)
        
        # 王炸判断
        if LandlordRules.is_royal_bomb(sorted_cards):
            return (LandlordRules.CardType.ROYAL_BOMB, 0, max_card)
        
        # 检查是否包含大小王（这些牌型不允许）
        has_joker = any(card.suite == Suite.JOKER for card in sorted_cards)
        if has_joker:
            return (LandlordRules.CardType.INVALID, 0, None)
        
        # 单张
        if card_count == 1:
            return (LandlordRules.CardType.SINGLE, 1, max_card)
        
        face_counts = LandlordRules.get_face_counts(sorted_cards)
        values = sorted(list(face_counts.values()))  # 排序后的计数列表
        unique_faces = sorted(face_counts.keys())    # 排序后的点数列表
        
        # 对子
        if card_count == 2 and values == [2]:
            return (LandlordRules.CardType.PAIR, 2, max_card)
        
        # 三张
        if card_count == 3 and values == [3]:
            return (LandlordRules.CardType.TRIPLE, 3, max_card)
        
        # 炸弹（4张相同）
        if card_count == 4 and values == [4]:
            return (LandlordRules.CardType.BOMB, max_card.face, max_card)
        
        # 顺子
        if len(unique_faces) == card_count and card_count >= 5:
            faces = [card.face for card in sorted_cards]
            if 2 not in faces and (max(faces) - min(faces) == card_count - 1):
                return (LandlordRules.CardType.STRAIGHT, card_count, max_card)
        
        # 连对
        if len(unique_faces) == card_count // 2 and card_count >= 6 and card_count % 2 == 0:
            if all(v == 2 for v in values):
                faces = sorted(unique_faces)
                if 2 not in faces and (max(faces) - min(faces) == len(faces) - 1):
                    return (LandlordRules.CardType.STRAIGHT_PAIR, len(faces), max_card)
        
        # 三带一
        if card_count == 4 and values == [1, 3]:
            return (LandlordRules.CardType.TRIPLE_WITH_SINGLE, 4, max_card)
        
        # 三带二
        if card_count == 5 and values == [2, 3]:
            return (LandlordRules.CardType.TRIPLE_WITH_PAIR, 5, max_card)
        
        # 6带2（总牌数8张：2组3张 + 2张单牌或1对）
        if card_count == 8:
            # 主牌必须是2组3张（计数包含两个3）
            if values.count(3) == 2:
                # 剩余2张可以是：2张单牌（[1,1,3,3]）或1对（[2,3,3]）
                if values in ([1, 1, 3, 3], [2, 3, 3]):
                    # 提取主牌点数（3张的点数）
                    triple_faces = [face for face, cnt in face_counts.items() if cnt == 3]
                    # 附属牌点数必须与主牌不同
                    sub_faces = [face for face, cnt in face_counts.items() if cnt != 3]
                    if not set(sub_faces) & set(triple_faces):  # 无交集
                        return (LandlordRules.CardType.SIX_WITH_TWO, max(triple_faces), max_card)
        
        # 6带4（总牌数10张：2组3张 + 2对对子）
        if card_count == 10:
            # 主牌2组3张，附属牌2对（计数为[2,2,3,3]）
            if values == [2, 2, 3, 3]:
                triple_faces = [face for face, cnt in face_counts.items() if cnt == 3]
                pair_faces = [face for face, cnt in face_counts.items() if cnt == 2]
                # 附属牌点数必须与主牌不同
                if not set(pair_faces) & set(triple_faces):
                    return (LandlordRules.CardType.SIX_WITH_FOUR, max(triple_faces), max_card)
        
        # 4带2（总牌数6张：1组4张炸弹 + 2张单牌或1对）
        if card_count == 6:
            # 主牌是1组4张炸弹
            if values.count(4) == 1:
                # 剩余2张可以是：2张单牌（[1,1,4]）或1对（[2,4]）
                if values in ([1, 1, 4], [2, 4]):
                    bomb_face = [face for face, cnt in face_counts.items() if cnt == 4][0]
                    sub_faces = [face for face, cnt in face_counts.items() if cnt != 4]
                    # 附属牌点数必须与炸弹不同
                    if not set(sub_faces) & {bomb_face}:
                        return (LandlordRules.CardType.FOUR_WITH_TWO, bomb_face, max_card)
        
        return (LandlordRules.CardType.INVALID, 0, None)
    
    @staticmethod
    def is_valid_play(cards: List[Card]) -> bool:
        """判断出牌是否为有效牌型"""
        card_type, _, _ = LandlordRules.get_card_type(cards)
        return card_type != LandlordRules.CardType.INVALID
    
    @staticmethod
    def can_beat(prev_cards: List[Card], curr_cards: List[Card]) -> bool:
        """判断当前牌能否压过上一手牌"""
        if not prev_cards:
            return LandlordRules.is_valid_play(curr_cards)
        
        prev_type, prev_key, prev_max = LandlordRules.get_card_type(prev_cards)
        curr_type, curr_key, curr_max = LandlordRules.get_card_type(curr_cards)
        
        if prev_type == LandlordRules.CardType.INVALID or curr_type == LandlordRules.CardType.INVALID:
            return False
        
        # 王炸压制一切
        if curr_type == LandlordRules.CardType.ROYAL_BOMB:
            return True
        
        # 上一手是王炸，无法压制
        if prev_type == LandlordRules.CardType.ROYAL_BOMB:
            return False
        
        # 炸弹系列压制逻辑（普通炸弹、4带2）
        if curr_type in (LandlordRules.CardType.BOMB, LandlordRules.CardType.FOUR_WITH_TWO):
            # 4带2属于炸弹衍生牌型，按炸弹规则比较
            if prev_type not in (LandlordRules.CardType.BOMB, LandlordRules.CardType.FOUR_WITH_TWO):
                return True
            # 与其他炸弹类比较时，比主牌大小
            return curr_key > prev_key
        
        # 对方是炸弹类，当前非炸弹类无法压制
        if prev_type in (LandlordRules.CardType.BOMB, LandlordRules.CardType.FOUR_WITH_TWO):
            return False
        
        # 相同牌型才能比较
        if prev_type != curr_type:
            return False
        
        # 顺子/连对需要长度相同
        if prev_type in [LandlordRules.CardType.STRAIGHT, LandlordRules.CardType.STRAIGHT_PAIR]:
            if prev_key != curr_key:
                return False
        
        # 6带2/6带4/4带2比较主牌最大点数
        if prev_type in [LandlordRules.CardType.SIX_WITH_TWO, 
                         LandlordRules.CardType.SIX_WITH_FOUR,
                         LandlordRules.CardType.FOUR_WITH_TWO]:
            return curr_key > prev_key
        
        # 其他牌型比较最大牌
        return curr_max > prev_max


# 测试新牌型
if __name__ == "__main__":
    # 6带2测试（2组3张+1对）
    c3 = [Card(Suite.SPADE, 3)]*3  # 333
    c4 = [Card(Suite.HEART, 4)]*3  # 444
    pair5 = [Card(Suite.CLUB, 5)]*2  # 55（附属牌）
    six_with_two = c3 + c4 + pair5  # 333444+55（8张）
    print("6带2（333444+55）判断：", LandlordRules.get_card_type(six_with_two)[0])  # 应返回SIX_WITH_TWO
    
    # 6带4测试（2组3张+2对）
    pair6 = [Card(Suite.DIAMOND, 6)]*2  # 66
    six_with_four = c3 + c4 + pair5 + pair6  # 333444+55+66（10张）
    print("6带4（333444+55+66）判断：", LandlordRules.get_card_type(six_with_four)[0])  # 应返回SIX_WITH_FOUR
    
    # 4带2测试（炸弹+2单）
    bomb3 = [Card(Suite.SPADE, 3)]*4  # 3333
    single5 = [Card(Suite.CLUB, 5), Card(Suite.DIAMOND, 6)]  # 5、6（单牌）
    four_with_two_1 = bomb3 + single5  # 3333+5+6（6张）
    print("4带2（3333+5+6）判断：", LandlordRules.get_card_type(four_with_two_1)[0])  # 应返回FOUR_WITH_TWO
    
    # 4带2测试（炸弹+1对）
    pair7 = [Card(Suite.HEART, 7)]*2  # 77
    four_with_two_2 = bomb3 + pair7  # 3333+77（6张）
    print("4带2（3333+77）判断：", LandlordRules.get_card_type(four_with_two_2)[0])  # 应返回FOUR_WITH_TWO
    
    # 压制测试
    bomb4 = [Card(Suite.SPADE, 4)]*4  # 4444
    stronger_four_with_two = bomb4 + pair7  # 4444+77
    print("4带2压制测试：", LandlordRules.can_beat(four_with_two_2, stronger_four_with_two))  # 应返回True
