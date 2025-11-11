# 1. 文档说明
"""
数据处理
"""

# 2. 导入
import os
from modules.order import Task

# 3. 全局变量
"""
os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
__file__：Python 的内置变量，代表当前代码所在文件的路径
os.path.abspath(__file__)：将 __file__ 转换为从系统根目录开始的完整路径（绝对路径）
os.path.dirname(...)：去掉文件名保留目录
os.path.dirname(...)：获得更上层文件目录地址
"""
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "task.txt")
# os.path.join(a, b, c)：Python 提供的 “路径拼接函数”，会自动根据操作系统补充路径分隔符

# 4. 函数
# ============================== 保存任务 ==============================
def save_task(task, file_path = file_path):
    """
    将单个任务追加保存到文本文件
    :param task    : Task 对象
    :param file_path: 任务存储文件路径(默认 data/task.txt)
    """
    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(f'{task.title}, {task.description}, {task.due_time}, {task.created_time}, {str(task.completed)}\n')
    except Exception as e:
        print(f'保存任务失败:{e}')
def save_all_tasks(tasks, file_path = file_path):
    """
    保存所有任务到文本文件（覆盖写入）
    :param tasks: Task对象列表
    :param file_path: 任务存储文件路径(默认 data/task.txt)
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for task in tasks:
                file.write(f'{task.title}, {task.description}, {task.due_time}, {task.created_time}, {str(task.completed)}\n')
    except Exception as e:
        print(f"错误:{e}")
# ============================== 读取任务 ==============================
def load_task(file_path = file_path):
    """
    从文本文件读取所有任务
    :param file_path: 任务存储文件路径(默认 data/task.txt)
    :return: 任务列表
    """
    tasks = []
    try:
        with open(file_path, 'r', encoding = 'utf-8') as file:
            # 使用enumerate获取行号，修复解包错误
            for line_num, line in enumerate(file, start=1):
                line = line.strip() # 去除行首尾的空格、换行符
                if not line:  # 跳过空行
                    continue
                data_line = line.split(', ') # 分割后得到字段列表
                if len(data_line) == 5: # 检验字段
                    title = data_line[0]
                    description = data_line[1]
                    due_time = data_line[2]
                    created_time = data_line[3]
                    # 将字符串转换为布尔值
                    completed = data_line[4].lower() == 'true' # completed 如果是true则返回true,如果是false返回false
                else:
                    print(f"警告：第{line_num}行数据格式错误（字段数量不符），已跳过该行")
                    continue
                task = Task(
                    title=title,
                    description=description,
                    due_time=due_time,
                    created_time=created_time,
                    completed=completed
                )
                tasks.append(task)
    except FileNotFoundError:
        print(f"提示：任务文件'{file_path}'不存在，将创建新文件")
    # 处理文件读取权限不足、文件损坏等IO错误
    except IOError as e:
        print(f"错误：读取任务文件失败，原因：{str(e)}")
    # 捕获其他未预料的错误
    except Exception as e:
        print(f"错误：加载任务时发生意外错误，原因：{str(e)}")
    return tasks
