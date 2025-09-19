import datetime
import time
from apscheduler.schedulers.background import BackgroundScheduler  # 改用后台调度器

# 导入你已实现的任务处理功能
from pytodo.models import Task
from pytodo.manager import (
    add_task, edit_task, view_tasks, complete_task,
    load_tasks_from_file
)

# ==================== 定时提醒功能 ====================
def check_due_tasks():
    """定时检查即将到期的任务并提醒"""
    tasks = load_tasks_from_file()
    if not tasks:
        return

    today = datetime.date.today()
    upcoming_tasks = []

    # 检查未来3天内到期的未完成任务
    for task in tasks:
        if not task.completed:
            try:
                due_date = datetime.datetime.strptime(task.due_date, "%Y-%m-%d").date()
                days_diff = (due_date - today).days
                
                if 0 <= days_diff <= 3:
                    upcoming_tasks.append((days_diff, task))
            except ValueError:
                continue  # 跳过日期格式错误的任务

    # 显示提醒
    if upcoming_tasks:
        print("\n" + "="*30)
        print("📅 任务到期提醒")
        print("="*30)
        for days, task in sorted(upcoming_tasks):
            status = "今天到期" if days == 0 else f"{days}天后到期"
            print(f"- {task.title} ({status})")
        print("="*30 + "\n")

# ==================== 主程序入口 ====================
def main():
    """程序主入口，处理用户交互和调度器启动"""
    print("="*40)
    print("欢迎使用任务管理系统 📝")
    print("="*40)

    # 初始化并启动定时任务调度器
    scheduler = BackgroundScheduler()
    # 每天上午9点检查一次任务
    scheduler.add_job(check_due_tasks, 'cron', hour=9, minute=0)
    # 为了测试方便，添加每分钟检查一次的任务
    scheduler.add_job(check_due_tasks, 'interval', minutes=1)
    scheduler.start()

    try:
        # 主交互循环
        while True:
            print("\n请选择操作：")
            print("1. 添加新任务")
            print("2. 查看任务列表")
            print("3. 编辑任务")
            print("4. 标记任务为完成")
            print("5. 退出程序")
            
            choice = input("请输入操作序号（1-5）：").strip()
            
            if choice == "1":
                add_task()
            elif choice == "2":
                view_tasks()
            elif choice == "3":
                edit_task()
            elif choice == "4":
                complete_task()
            elif choice == "5":
                print("感谢使用，再见！")
                break
            else:
                print("无效的选择，请重新输入")
            
            # 每次操作后暂停一下，让用户看清结果
            time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        # 优雅关闭调度器
        scheduler.shutdown()
        print("\n程序已退出")

if __name__ == "__main__":
    main()
