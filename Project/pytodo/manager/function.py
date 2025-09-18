import datetime
import time
from apscheduler.schedulers.blocking import BlockingScheduler

from pytodo.models import Task

# ==================== 创建任务 ====================

def add_task():
    """
    命令行交互创建新任务，自动生成创建时间
    """

    # 接收用户输入

    title = input("请输入任务标题：")
    description = input("请输入任务描述：")
    due_date = input("请输入任务截止日期（格式：YYYY-MM-DD）：")

    # 生成当前日期作为创建时间（格式统一）

    created_at = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # 验证截止日期格式（基础校验）

    try:
        datetime.datetime.strptime(due_date, "%Y-%m-%d")
    except ValueError:
        print("错误：截止日期格式无效，请重新输入（示例：2024-12-31）")
        return
    
    # 创建 Task 对象并保存

    new_task = Task(title, description, due_date, created_at)
    save_task_to_file(new_task)
    print(f"任务创建成功：{new_task}")

# ==================== 保存任务到文件 ====================

def save_task_to_file(task, filepath="../tasks.txt"):
    """
    将单个任务追加保存到文本文件
    :param task: Task 对象
    :param filepath: 任务存储文件路径（默认 tasks.txt）
    """
    try:
        with open(filepath, "a", encoding="utf-8") as file:

            # 按格式写入字段（completed 转换为字符串便于存储）

            file.write(f"{task.title}, {task.description}, {task.due_date}, {task.created_at}, {str(task.completed)}\n")
    except Exception as e:
        print(f"保存任务失败：{e}")

def save_all_tasks(tasks, filepath="../tasks.txt"):
    """
    覆盖保存所有任务到文件（用于编辑/删除后更新全量任务）
    :param tasks: Task 对象列表
    :param filepath: 任务存储文件路径（默认 tasks.txt）
    """
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            for task in tasks:
                file.write(f"{task.title}, {task.description}, {task.due_date}, {task.created_at}, {str(task.completed)}\n")
    except Exception as e:
        print(f"全量保存任务失败：{e}")
# =================================================================================
def load_tasks_from_file(file_path="../tasks.txt"):
    """
    从任务存储文件中读取数据，还原为Task实例列表
    
    参数：
        file_path: 任务文件路径，默认是文档中约定的"tasks.txt"
    返回：
        list[Task]: 包含所有任务的Task实例列表；若文件不存在/读取失败，返回空列表
    异常处理：
        - 文件不存在：返回空列表（首次使用工具时无文件，避免崩溃）
        - 文件读取权限不足：捕获IOError并提示，返回空列表
        - 数据格式错误（如字段缺失、状态值非法）：跳过错误行并警告，继续加载其他任务
    """
    tasks = []  # 最终返回的Task实例列表
    try:
        # 1. 打开文件（只读模式，编码为utf-8避免中文乱码）
        with open(file_path, "r", encoding="utf-8") as file:
            # 2. 逐行读取文件（跳过空行，避免解析无效数据）
            for line_num, line in enumerate(file, start=1):  # line_num：记录行号，便于定位错误
            # enumerate()给 “可迭代对象”的每个元素，自动添加一个 “索引编号”
                line = line.strip()  # 去除行首尾的空格、换行符
                if not line:  # 跳过空行
                    continue

                # 3. 解析行数据（按文档约定的"逗号+空格"分割字段）
                # 文档中存储格式示例："Task 1, Complete project, 2023-04-10, 2023-04-01, False"
                fields = line.split(", ")  # 分割后得到5个字段的列表

                # 4. 校验字段数量（必须为5个，否则格式错误）
                if len(fields) != 5:
                    print(f"警告：第{line_num}行数据格式错误（字段数量不足），已跳过该行")
                    continue

                # 5. 提取字段并转换数据类型
                title = fields[0]
                description = fields[1]
                due_date = fields[2]
                created_at = fields[3]
                # 完成状态：将字符串"True"/"False"转为布尔值（兼容大小写，如"true"/"TRUE"）
                completed_str = fields[4].strip().lower()
                if completed_str not in ["true", "false"]:
                    print(f"警告：第{line_num}行'完成状态'值非法（需为True/False），已跳过该行")
                    continue
                completed = (completed_str == "true")  # 转为布尔值

                # 6. 创建Task实例并添加到列表
                task = Task(
                    title=title,
                    description=description,
                    due_date=due_date,
                    created_at=created_at,
                    completed=completed
                )
                tasks.append(task)

    # 处理文件不存在的情况（首次使用工具时正常）
    except FileNotFoundError:
        print(f"提示：任务文件'{file_path}'不存在，将创建新文件（当前任务列表为空）")
    # 处理文件读取权限不足、文件损坏等IO错误
    except IOError as e:
        print(f"错误：读取任务文件失败，原因：{str(e)}（当前任务列表为空）")
    # 捕获其他未预料的错误
    except Exception as e:
        print(f"错误：加载任务时发生意外错误，原因：{str(e)}（当前任务列表为空）")

    # 返回最终加载的任务列表（空列表或有效Task实例列表）
    return tasks

# =================================================================================

# 编辑任务

def edit_task():
    """通过序号（索引）编辑任务"""
    # 1. 加载所有任务并检查是否为空
    tasks = load_tasks_from_file()
    if not tasks:
        print("当前没有任务可编辑")
        return

    # 2. 显示所有任务及序号（方便用户选择）
    print("\n===== 所有任务 =====")
    for idx, task in enumerate(tasks):  # idx 就是序号（从0开始）
        print(f"序号 {idx}：{task}")  # 显示序号+任务信息
    print("===================")

    # 3. 让用户输入要编辑的序号
    try:
        task_idx = int(input("请输入要编辑的任务序号："))
        # 验证序号有效性（必须在0到任务数量-1之间）
        if not (0 <= task_idx < len(tasks)):
            print(f"错误：序号必须在 0 到 {len(tasks)-1} 之间")
            return
    except ValueError:
        print("错误：请输入有效的数字序号")
        return

    # 4. 获取要编辑的任务对象
    task = tasks[task_idx]
    print(f"\n当前编辑的任务：{task}")

    # 5. 交互修改任务属性（支持保留原内容，直接回车不修改）
    new_title = input(f"请输入新标题（原：{task.title}）：")
    new_description = input(f"请输入新描述（原：{task.description}）：")
    new_due_date = input(f"请输入新截止日期（原：{task.due_date}，格式YYYY-MM-DD）：")
    new_completed = input(f"请输入完成状态（原：{task.completed}，输入True/False）：")

    # 6. 更新属性（只更新用户输入了内容的字段）
    if new_title:  # 如果用户输入了内容（非空）
        task.title = new_title
    if new_description:
        task.description = new_description
    # 验证并更新截止日期
    if new_due_date:
        try:
            datetime.datetime.strptime(new_due_date, "%Y-%m-%d")
            task.due_date = new_due_date
        except ValueError:
            print("警告：截止日期格式无效，未更新")
    # 验证并更新完成状态
    if new_completed:
        new_completed = new_completed.lower()
        if new_completed in ["true", "false"]:
            task.completed = (new_completed == "true")
        else:
            print("警告：完成状态格式无效（需True/False），未更新")

    # 7. 覆盖保存所有任务（关键：确保修改后的数据替换原文件）
    save_all_tasks(tasks)
    print(f"任务编辑成功：{task}")

# ==================== 查看任务 ====================

# 排序函数（已提供）
def sort_tasks(tasks, sort_by="created_at"):
    """任务排序（支持按创建时间/截止日期）"""
    if sort_by == "due_date":
        return sorted(tasks, key=lambda x: x.due_date)
    elif sort_by == "created_at":
        return sorted(tasks, key=lambda x: x.created_at)
    else:
        print(f"警告：不支持的排序字段 {sort_by}，使用默认排序（创建时间）")
        return tasks

# 过滤函数（已提供）
def filter_tasks(tasks, filter_by=None):
    """任务过滤（支持按完成状态）"""
    if filter_by == "completed":
        return [task for task in tasks if task.completed]
    elif filter_by == "uncompleted":
        return [task for task in tasks if not task.completed]
    else:
        return tasks  # 无过滤时返回全部

# 查看任务（整合排序和过滤）
def view_tasks():
    # 1. 加载所有任务
    tasks = load_tasks_from_file()
    if not tasks:
        print("当前没有任务可查看")
        return

    # 2. 交互：让用户选择过滤条件
    print("\n===== 过滤选项 =====")
    print("1. 全部任务")
    print("2. 仅显示已完成任务")
    print("3. 仅显示未完成任务")
    filter_choice = input("请选择过滤方式（输入序号1-3）：").strip()
    # 映射用户选择到过滤参数（对应filter_tasks的filter_by参数）
    filter_map = {
        "1": None,          # 无过滤
        "2": "completed",   # 已完成
        "3": "uncompleted"  # 未完成
    }
    filter_by = filter_map.get(filter_choice, None)  # 无效输入默认显示全部
    filtered_tasks = filter_tasks(tasks, filter_by)

    if not filtered_tasks:
        print("没有符合条件的任务")
        return

    # 3. 交互：让用户选择排序方式
    print("\n===== 排序选项 =====")
    print("1. 按创建时间排序（默认）")
    print("2. 按截止日期排序")
    sort_choice = input("请选择排序方式（输入序号1-2）：").strip()
    # 映射用户选择到排序参数（对应sort_tasks的sort_by参数）
    sort_map = {
        "1": "created_at",  # 创建时间
        "2": "due_date"     # 截止日期
    }
    sort_by = sort_map.get(sort_choice, "created_at")  # 无效输入默认按创建时间
    sorted_tasks = sort_tasks(filtered_tasks, sort_by)

    # 4. 显示处理后的任务（带序号，方便后续编辑）
    print("\n===== 任务列表 =====")
    for idx, task in enumerate(sorted_tasks):
        print(f"序号 {idx}：{task}")
    print("===================")
    print(f"提示：当前显示{get_filter_desc(filter_by)}，按{get_sort_desc(sort_by)}排序")
    print(f"编辑任务时，请使用以上序号")

# 辅助函数：获取过滤条件的描述文字（用于显示提示）
def get_filter_desc(filter_by):
    if filter_by == "completed":
        return "【已完成任务】"
    elif filter_by == "uncompleted":
        return "【未完成任务】"
    else:
        return "【全部任务】"
# 辅助函数：获取排序方式的描述文字（用于显示提示）
def get_sort_desc(sort_by):
    if sort_by == "due_date":
        return "截止日期"
    else:
        return "创建时间"

# ==================== 确认任务完成 ====================

def complete_task():
    """通过序号快速标记任务为完成（简化流程）"""
    tasks = load_tasks_from_file()
    if not tasks:
        print("当前没有任务可标记完成")
        return

    # 只显示未完成任务（更符合使用场景）
    uncompleted_tasks = filter_tasks(tasks, "uncompleted")
    if not uncompleted_tasks:
        print("所有任务都已完成！")
        return

    # 按截止日期排序（优先处理快到期的任务）
    sorted_tasks = sort_tasks(uncompleted_tasks, "due_date")
    
    print("\n===== 可标记完成的任务（未完成） =====")
    for idx, task in enumerate(sorted_tasks):
        print(f"序号 {idx}：{task}")
    print("=====================================")

    try:
        task_idx = int(input("请输入要标记完成的任务序号："))
        if not (0 <= task_idx < len(sorted_tasks)):
            print(f"错误：序号必须在 0 到 {len(sorted_tasks)-1} 之间")
            return
    except ValueError:
        print("错误：请输入有效的数字序号")
        return

    # 标记为完成并保存
    task = sorted_tasks[task_idx]
    task.completed = True
    # 保存全量任务（注意：需要更新原始任务列表，而非过滤后的列表）
    # 找到原始任务在全量列表中的位置并更新
    for t in tasks:
        if (t.title == task.title and t.created_at == task.created_at):
            t.completed = True
            break
    save_all_tasks(tasks)
    print(f"任务已标记为完成：{task}")

# ==================== 系统提醒 ====================
