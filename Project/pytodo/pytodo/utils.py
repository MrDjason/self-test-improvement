import datetime
from modules.order import Task
# ============================== 保存任务 ==============================
from data.data_handler import save_task
# ============================== 创建任务 ==============================
def add_task():
    title = input('请输入标题:')
    description = input('请输入描述:')
    due_time = input('请输入任务截至时间(格式为YYYY-MM-DD-HH:mm):')
    created_time = datetime.datetime.now().strftime("%Y-%m-%d")

    try:
        datetime.datetime.strptime(due_time, '%Y-%m-%d-%H:%M')
    except ValueError:
        print("错误：截止时间格式无效，请重新输入(示例:2024-12-31-12:59)")
        return
    
    new_task = Task(title, description, due_time, created_time)
    save_task(new_task)
# ============================== 查看任务 ==============================

# ============================== 删除任务 ==============================
# ============================== 编辑任务 ==============================
# ============================== 确认完成 ==============================
