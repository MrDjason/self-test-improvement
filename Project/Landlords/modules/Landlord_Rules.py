# ==================== 导入模块 ====================
import random
from enum import Enum
from typing import List, Tuple, Optional

# 导入扑克模型中的类
from modules.poker_model import Suite, Card

class LandlordRules:
    """斗地主出牌规则类"""
    
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
        SIX_WITH_TWO = 8  # 6带2（2组3张+2张单牌或1对）
        SIX_WITH_FOUR = 9  # 6带4（2组3张+2对对子）
        FOUR_WITH_TWO = 10  # 4带2（炸弹+2张单牌或1对）
        BOMB = 11      # 炸弹(4张相同)
        ROYAL_BOMB = 12  # 王炸(大小王)

    @staticmethod
    # @staticmethod 装饰器
    # 不需要创建类的实例，直接通过类名.方法名()调用,
    def get_face_counts(cards: List[Card]) -> dict:
    # cards: List[Card] 给传入参数做类型注解，表示是cards是列表，并且传入的元素都是Card对象
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
        # 如果没有上一手牌，则只需要判断当前牌是否有效
        if not prev_cards:
            return LandlordRules.is_valid_play(curr_cards)
        
        # 获取牌型信息
        prev_type, prev_key, prev_max = LandlordRules.get_card_type(prev_cards)
        curr_type, curr_key, curr_max = LandlordRules.get_card_type(curr_cards)
        
        # 检查牌型是否有效
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
        
        # 6带2/6带4比较主牌最大点数
        if prev_type in [LandlordRules.CardType.SIX_WITH_TWO, LandlordRules.CardType.SIX_WITH_FOUR]:
            return curr_key > prev_key
        
        # 其他牌型比较最大牌
        return curr_max > prev_max
        
    @staticmethod
    def get_card_type_name(card_type: 'LandlordRules.CardType') -> str:
        """获取牌型的中文名称"""
        type_names = {
            LandlordRules.CardType.SINGLE: "单张",
            LandlordRules.CardType.PAIR: "对子",
            LandlordRules.CardType.TRIPLE: "三张",
            LandlordRules.CardType.STRAIGHT: "顺子",
            LandlordRules.CardType.STRAIGHT_PAIR: "连对",
            LandlordRules.CardType.TRIPLE_WITH_SINGLE: "三带一",
            LandlordRules.CardType.TRIPLE_WITH_PAIR: "三带二",
            LandlordRules.CardType.SIX_WITH_TWO: "6带2",
            LandlordRules.CardType.SIX_WITH_FOUR: "6带4",
            LandlordRules.CardType.FOUR_WITH_TWO: "4带2",
            LandlordRules.CardType.BOMB: "炸弹",
            LandlordRules.CardType.ROYAL_BOMB: "王炸",
            LandlordRules.CardType.INVALID: "无效牌型"
        }
        return type_names.get(card_type, "未知牌型")
        
    @staticmethod
    def get_available_moves(player_cards: List[Card], last_cards: List[Card] = None) -> List[List[Card]]:
        """获取玩家当前可以出的所有有效牌型"""
        available_moves = []
        
        # 如果没有上一手牌，则所有有效牌型都可以出
        if not last_cards:
            # 这里实现一个简化版：返回所有可能的有效单牌、对子、三张
            # 在实际游戏中可以实现更复杂的组合
            
            # 获取所有单牌
            for i in range(len(player_cards)):
                single_card = [player_cards[i]]
                if LandlordRules.is_valid_play(single_card):
                    available_moves.append(single_card)
            
            # 获取所有对子
            for i in range(len(player_cards)):
                for j in range(i + 1, len(player_cards)):
                    pair = [player_cards[i], player_cards[j]]
                    if LandlordRules.is_valid_play(pair):
                        available_moves.append(pair)
            
            # 获取所有三张
            for i in range(len(player_cards)):
                for j in range(i + 1, len(player_cards)):
                    for k in range(j + 1, len(player_cards)):
                        triple = [player_cards[i], player_cards[j], player_cards[k]]
                        if LandlordRules.is_valid_play(triple):
                            available_moves.append(triple)
        else:
            # 有上一手牌时，需要能压过才能出
            # 为了简化，这里只返回所有可能的炸弹
            # 检查是否有王炸
            jokers = [card for card in player_cards if card.suite == Suite.JOKER]
            if len(jokers) == 2:
                royal_bomb = jokers
                if LandlordRules.can_beat(last_cards, royal_bomb):
                    available_moves.append(royal_bomb)
            
            # 检查是否有普通炸弹
            face_counts = {}  # 统计每个点数出现的次数
            for card in player_cards:
                if card.suite != Suite.JOKER:  # 排除大小王
                    key = card.face
                    face_counts[key] = face_counts.get(key, []) + [card]
            
            for face, cards in face_counts.items():
                if len(cards) >= 4:
                    bomb = cards[:4]  # 取前4张作为炸弹
                    if LandlordRules.can_beat(last_cards, bomb):
                        available_moves.append(bomb)
        
        # 去重并返回结果
        # 由于列表无法直接哈希，这里采用一种简单的去重方式
        unique_moves = []
        for move in available_moves:
            if move not in unique_moves:
                unique_moves.append(move)
        
        return unique_moves
