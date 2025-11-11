import datetime
from modules.order import Task
from data.data_handler import *

# ============================== 保存任务 ==============================
# from .data.data_handler import save_task
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
def view_task():
    tasks = load_task()
    if not tasks:
        print('错误:当前tasks空')
        return
        # 执行 return（空的），直接结束 view_task 函数
    print(f'{'=' * 18} 任务 {'=' * 18}')
    # 使用enumerate添加索引
    for index, task in enumerate(tasks):
    # enumerate(tasks) 会生成类似 (0, 任务1), (1, 任务2)的元组
    # 循环中通过 “解包” 获取元组的两个值：index, task
        print(f'序号{index+1}:{task}')
        # 将 Task 对象传给 print 函数,自动调用该对象的 __str__ 方法(或 __repr__ 方法)
    print(f'{'=' * 18} 结束 {'=' * 18}')
    return tasks

# ============================== 删除任务 ==============================
def del_task():
    tasks = view_task()
    try:
        task_index = int(input('请输入你需要删除的任务id:'))
        if not (1<= task_index <=len(tasks)):
            print(f"错误:序号必须在 1 到 {len(tasks)} 之间")
            return
    except ValueError:
        print('错误:请输入有效数字')
        return
    
    # 处理流程
    deleted_task = tasks[task_index - 1] # index从1开始,需要减去
    confirm = input(f"确定要删除任务 '{deleted_task.title}' 吗？(y/n):").lower().strip()
    if confirm != 'y':
        print("已取消删除操作")
        return
    tasks.pop(task_index - 1)
    save_all_tasks(tasks)
    print(f"任务已成功删除:{deleted_task.title}")

# ============================== 编辑任务 ==============================
def edit_task():
    tasks = view_task()
    try:
        task_index = int(input('请输入你需要编辑的任务id:'))
        if not (1<= task_index <=len(tasks)):
            print(f"错误:序号必须在 1 到 {len(tasks)} 之间")
            return
    except ValueError:
        print('错误:请输入有效数字')
        return
    
    task = tasks[task_index - 1]
    print(f"\n当前编辑的任务:{task}")

    # 交互修改任务属性（支持保留原内容，直接回车不修改）
    new_title = input(f"请输入新标题(原：{task.title}):")
    new_description = input(f"请输入新描述(原：{task.description}):")
    new_due_time = input(f"请输入新截止日期(原：{task.due_time},格式YYYY-MM-DD-HH-mm):")
    new_completed = input(f"请输入完成状态(原：{task.completed},输入True/False):")

    # 更新属性(只更新用户输入了内容的字段)
    if new_title:  # 如果用户输入了内容(非空)
        task.title = new_title
    if new_description:
        task.description = new_description
    # 验证并更新截止日期
    if new_due_time:
        try:
            datetime.datetime.strptime(new_due_time, '%Y-%m-%d-%H:%M')
            task.due_time = new_due_time
        except ValueError:
            print('错误:截止时间无效')
    if new_completed:
        new_completed = new_completed.lower()
        if new_completed in ['true', 'false']: # new_completed 如果是true则返回true,如果是false返回false
            task.completed = new_completed
        else:
            print('错误:完成状态未设置成功')
    
    print(f"任务已成功编辑:{task.title}")
    save_all_tasks(tasks)

# ============================== 确认完成 ==============================
def complete_task():
    """
    标记任务为已完成状态
    通过用户输入的任务ID查找任务，并将其completed属性设置为True
    更新后保存到文件中
    """
    tasks = load_task()
    
    # 如果没有任务，直接返回
    if not tasks:
        print('当前没有任务')
        return
    
    # 过滤未完成项
    uncompleted = []
    for task in tasks:
        if not task.completed: # task是类Task的对象
            uncompleted.append(task)
    
    # 如果所有任务都已完成，提示用户
    if not uncompleted:
        print('所有任务都已完成!')
        return

    print(f'{'=' * 17} 未完成项 {'=' * 17}')
    for index, task in enumerate(uncompleted):
        print(f'序号{index + 1}:{task}')

    try:
        task_index = int(input('请输入你需要完成的任务id:'))
        if not (1<= task_index <= len(uncompleted)):
            print(f"错误:序号必须在 1 到 {len(uncompleted)} 之间")
            return
    except ValueError:
        print('错误:请输入有效数字')
        return
    
    # 获取用户选择的未完成任务
    selected_task = uncompleted[task_index - 1]
    
    # 在原始任务列表中找到并更新该任务
    for task in tasks:
        # 这里假设标题、描述和截止时间的组合可以唯一标识一个任务
        if (task.title == selected_task.title and 
            task.description == selected_task.description and 
            task.due_time == selected_task.due_time):
            task.completed = True
            break
    
    save_all_tasks(tasks)  # 保存更新后的任务列表
    print(f"任务 '{selected_task.title}' 已标记为完成状态")
    