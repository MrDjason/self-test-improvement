import datetime
import time

# 导入你已实现的任务处理功能
from models.task import Task  # 导入Task类
from manager.function import (
    add_task, edit_task, view_tasks, complete_task, delete_task,
    load_tasks_from_file
)



# ==================== 主程序入口 ====================
def main():
    """程序主入口，处理用户交互"""
    print("="*40)
    print("欢迎使用任务管理系统 📝")
    print("="*40)

    try:
        # 主交互循环
        while True:
            print("\n请选择操作：")
            print("1. 添加新任务")
            print("2. 查看任务列表")
            print("3. 编辑任务")
            print("4. 标记任务为完成")
            print("5. 删除任务")
            print("6. 退出程序")
            
            choice = input("请输入操作序号（1-6）：").strip()
            
            if choice == "1":
                add_task()
            elif choice == "2":
                view_tasks()
            elif choice == "3":
                edit_task()
            elif choice == "4":
                complete_task()
            elif choice == "5":
                delete_task()
            elif choice == "6":
                print("感谢使用，再见！")
                break
            else:
                print("无效的选择，请重新输入")
            
            # 每次操作后暂停一下，让用户看清结果
            time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        print("\n程序已退出")

if __name__ == "__main__":
    main()
