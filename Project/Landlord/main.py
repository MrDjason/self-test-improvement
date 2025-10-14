# ==================== 导入模块 ====================

# ==================== 游    戏 ====================
def deal_cards():
    pass
def play_game():
    pass
def bid_landlord():
    pass
# ==================== 主 界 面 ====================
def main():
    while True:
        print('='*20 + '欢迎来到斗地主游戏！' + '='*20)
        print("1. 开始游戏")
        print("2. 退出游戏")
        choice = input('请输入你的选择(1或者2):')
        if choice == '1':
            pass
        elif choice =='2':
            print('感谢使用，再见！')
            break
        else:
            print("无效输入，请重新选择！")

# ==================== 运    行 ====================
if __name__ == "__main__":
    main()