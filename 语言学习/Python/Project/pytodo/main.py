# ============================== 引入库 ==============================
from 语言学习.Python.Project.pytodo.utils import *
import time
# ============================== 主程序 ==============================
def main():
    try:
        while True:
            print('=' * 40)
            print('欢迎使用任务管理系统 📝')
            print('=' * 40)
            print('1.添加任务')
            print('2.查看任务')
            print('3.删除任务')
            print('4.编辑任务')
            print('5.完成任务')
            print('6.结束')
            choice = input('请输入你的选项(1-6):').strip()
            if choice == '1':
                add_task()
            elif choice == '2':
                view_task()
            elif choice == '3':
                del_task()
            elif choice == '4':
                edit_task()
            elif choice == '5':
                complete_task()
            elif choice == "6":
                print("感谢使用，再见！")
                break
            else:
                print("无效的选择，请重新输入:")
            
            # 每次操作后暂停一下，让用户看清结果
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("\n程序已退出")
# ============================== 运行它 ==============================
if __name__ == '__main__':
    main()
    # 2025-09-30-12:00