# ==================== 导入模块 ====================
from enum import Enum
from typing import List, Tuple, Optional
from modules.poker_model import *

# ==================== 定 义 类 ====================
class LandLordRules():

    class CardType(Enum):
    # 定义牌型枚举类
        INVALID = 0             # 无效牌型
        SINGLE = 1              # 单张
        PAIR = 2                # 对子
        TRIPLE = 3              # 三张
        STRAIGHT = 4            # 顺子
        STRAIGHT_PAIR = 5       # 连对(3+)
        TRIPLE_WITH_SINGLE =  6 # 三带一
        TRIPLE_WITH_PAIR = 7    # 三带二
        SIX_WITH_TWO = 8        # 六带二
        SIX_WITH_FOUR = 9       # 六带四
        FOUR_WITH_TWO = 10      # 四带二
        BOMB = 11               # 炸弹
        ROYAL_BOMB = 12         # 王炸

    @staticmethod
    def get_card_type(cards: List[Card]) -> Tuple[CardType, Optional[Card], int]:
    # int 会针对性存储牌型的优先级，让后续函数能快速判断 “当前牌能否压过对方”
    # Optional[Card] 这个位置的返回值要么是 Card 类型的对象，要么是 None(返回值可能为空)
        if not cards:
            return (LandLordRules.CardType.INVALID, 0, None)

        sorted_cards = sorted(cards)
        max_card = sorted_cards[-1]
        card_count = len(sorted_cards)

        # 判断王炸
        if card_count == 2:
            if (sorted_cards[0].suite == Suite.JOKER and sorted_cards[1].suite== Suite.JOKER):
                return (LandLordRules.CardType.ROYAL_BOMB, None,2)

        # 判断王牌
        has_joker = any(card.suite == Suite.JOKER for card in sorted_cards)
        # any(...) 接受一个可迭代对象，迭代对象中有至少一个值为 True，any() 就返回 True
        if has_joker:
            if card_count == 1:
                return (LandLordRules.CardType.SINGLE, max_card, 1)
            else:
                return (LandLordRules.CardType.INVALID, None, 0) # 大小王只能走单张

        # 非王牌的牌型判断
        face_counts = {}
        for card in cards:
            key = card.face
            face_counts[key] = face_counts.get(key, 0) + 1
            # 获取{点数:次数}字典,查看各点数各出现多少次
            # 用字典的 get 方法获取 face_counts 中 key 对应的现有次数
            # 如果 key 已经在字典里，现有次数+1。如果 key 第一次统计该点数，0+1
        values = sorted(list(face_counts.values())) # 获取字典中所有的 “次数值”，并装入列表后排序
        unique_faces = sorted(face_counts.keys())   # 提取所有key，即“牌组中不重复的点数”后整理

        # 单张
        if card_count == 1:
            return (LandLordRules.CardType.SINGLE, max_card, 1)

        # 对子
        if card_count == 2 and values == [2]: # values == [2] 意味着字典只有一个重复了两次的元素
            return (LandLordRules.CardType.PAIR, max_card, 2)
        
        # 三张
        if card_count == 3 and values == [3]:
            return (LandLordRules.CardType.TRIPLE, max_card, 3)

        # 炸弹
        if card_count == 4 and values == [4]:
            return (LandLordRules.CardType.BOMB, max_card, 4)
        
        # 顺子
        if card_count >=5 and len(unique_faces) == card_count: # 不重复点数牌数等于总牌数
            faces = [card.face for card in sorted_cards]
            if 2 in faces:
                pass
            else:
                processed_faces = sorted([14 if face == 1 else face for face in faces])
                # processed_faces = []
                # for face in faces:
                # if face == 1:
                #   processed_faces.append(14)
                # else:
                #   processed_faces.append(face)
                if processed_faces[-1] - processed_faces[0] == card_count - 1:
                    return(LandLordRules.CardType.STRAIGHT, max_card, card_count)
                
        # 连对
        if card_count >=6 and len(unique_faces) == card_count // 2 and card_count % 2 == 0:
            if all(value == 2 for value in values): # 判断所有出现的值是否都重复了两次
                if 2 not in unique_faces and (max(unique_faces) - min(unique_faces) == len(unique_faces) - 1):
                # 55 66 77 -> 7-5 = 3-1
                    return (LandLordRules.CardType.STRAIGHT_PAIR, max_card, len(unique_faces))
                
        # 三带一
        if card_count == 4 and values == [1, 3]: # values 统计重复数字出现次数,经过排序从小到大显示
            triple_face = [face for face, count in face_counts.items() if count == 3][0]
            # triple_face =[]
            # for face, count in face_counts.items():
            #   if count == 3:
            #       triple_face.append(face)
            # triple_face = triple_face[0]
            return (LandLordRules.CardType.TRIPLE_WITH_SINGLE, triple_face, 4)
        
        # 三带二
        if card_count == 5 and values == [2, 3]:
            triple_face = [face for face, count in face_counts.items() if count == 3][0]
            return (LandLordRules.CardType.TRIPLE_WITH_PAIR, triple_face, 5)
        
        # 六带二
        if card_count == 8:
            if values.count(3) == 2 and values in ([1,1,3,3],[2,3,3]):
                triple_faces = sorted([face for face, count in face_counts.items() if count == 3])
                if (triple_faces[1]-triple_faces[0]==1):
                    return (LandLordRules.CardType.SIX_WITH_TWO, max(triple_faces), 8)

        # 六带四
        if card_count == 10 and values == [2, 2, 3, 3]:
                triple_faces = sorted([face for face, count in face_counts.items() if count == 3])
                if (triple_faces[1]-triple_faces[0]==1):
                    return (LandLordRules.CardType.SIX_WITH_FOUR, max(triple_faces), 10)
                
        # 四带二
        if card_count == 6 and values in ([1,1,4], [2, 4]):
            bomb_face = [face for face, count in face_counts.items() if count==4][0]
            return (LandLordRules.CardType.FOUR_WITH_TWO, bomb_face, 6)
        
    @staticmethod
    def is_valid(cards:List[Card]) -> bool:
        card_type, _, _ = LandLordRules.get_card_type(cards)
        return card_type != LandLordRules.CardType.INVALID
    
    @staticmethod
    def can_beat(previous_cards: List[Card],current_cards: List[Card]) -> bool:
        # 如果没有上一手牌，仅判断当前出牌是否有效
        if not previous_cards:
            return LandLordRules.is_valid(current_cards)
        
        # 获取牌型信息
        previous_type, previous_max, previous_num = LandLordRules.get_card_type(previous_cards)
        current_type, current_max, current_num = LandLordRules.get_card_type(current_cards)

        # 必须同类型才能比较
        if previous_type != current_type:
            return False
        
        # 必须同长度才能比较
        if previous_num != current_num:
            return False

        # 检查牌型是否有效
        if current_type == LandLordRules.CardType.INVALID:
            return False
        
        # 王炸大于一切
        if current_type == LandLordRules.CardType.ROYAL_BOMB:
            return True
        
        # 上一手是王炸，无法压制
        if previous_type == LandLordRules.CardType.ROYAL_BOMB:
            return False
        
        # 炸弹压制
        if current_type == LandLordRules.CardType.BOMB:
            if previous_type != LandLordRules.CardType.BOMB:
                return True
            elif previous_type == LandLordRules.CardType.BOMB:
                return current_max > previous_max # 比炸弹牌面大小
            
        # 上一手是炸弹
        if previous_type == LandLordRules.CardType.BOMB:
            return False
        
        # 顺子/连对需要长度相同
        if previous_type in [LandLordRules.CardType.STRAIGHT, LandLordRules.CardType.STRAIGHT_PAIR]:
            if previous_num != current_num:
                return False
            
        # 单牌、对子、三张、三带一、三带二、四带二、六带二、六带四 比较
        return current_max > previous_max
    
    @staticmethod
    def get_card_type_name(card_type: 'LandLordRules.CardType') -> str:
        '''获取牌型的中文名称'''
        type_names = {
            LandLordRules.CardType.SINGLE: "单张",
            LandLordRules.CardType.PAIR: "对子",
            LandLordRules.CardType.TRIPLE: "三张",
            LandLordRules.CardType.STRAIGHT: "顺子",
            LandLordRules.CardType.STRAIGHT_PAIR: "连对",
            LandLordRules.CardType.TRIPLE_WITH_SINGLE: "三带一",
            LandLordRules.CardType.TRIPLE_WITH_PAIR: "三带二",
            LandLordRules.CardType.SIX_WITH_TWO: "6带2",
            LandLordRules.CardType.SIX_WITH_FOUR: "6带4",
            LandLordRules.CardType.FOUR_WITH_TWO: "4带2",
            LandLordRules.CardType.BOMB: "炸弹",
            LandLordRules.CardType.ROYAL_BOMB: "王炸",
            LandLordRules.CardType.INVALID: "无效牌型"
        }
        return type_names.get(card_type, "未知牌型") 
        # card_type 在字典中存在，则返回对应的中文。不存在，则返回默认值 “未知牌型”

        