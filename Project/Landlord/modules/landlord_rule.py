# ==================== 导入模块 ====================
from enum import Enum
from typing import List, Tuple, Optional
from modules.poker_model import *

# ==================== 定 义 类 ====================
class LandLordRules():

    class CardType(Enum):
    # 定义牌型枚举类
        Invalid = 0             # 无效牌型
        Single = 1              # 单张
        Pair = 2                # 对子
        Triple = 3              # 三张
        Straight = 4            # 顺子
        Straight_Pair = 5       # 连对(3+)
        Triple_With_Single =  6 # 三带一
        Triple_With_Pair = 7    # 三带二
        Six_with_two = 8        # 六带二
        Six_with_four = 9       # 六带四
        Four_with_two = 10      # 四带二
        Bomb = 11               # 炸弹
        Royal_Bomb = 12         # 王炸

    @staticmethod
    def get_face_counts(cards: List[Card]) -> dict:
        pass

    @staticmethod

