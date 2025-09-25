from pytodo.modules.order import Task
# ============================== 保存任务 ==============================
def save_task(task, file_path='task.txt'):
    """
    将单个任务追加保存到文本文件
    :param task    : Task 对象
    :param filepath: 任务存储文件路径(默认 task.txt)
    """
    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(f'{task.title}, {task.description}, {task.due_time}, {task.created_time}, {task.completed}')
    except Exception as e:
        print(f'保存任务失败:{e}')

# ============================== 读取任务 ==============================
def load_task(file_path = 'task.txt'):
    tasks = []
    try:
        with open(file_path, 'r', encoding = 'utf-8') as file:
            for line_num, line in file:
                line = line.strip() # 去除行首尾的空格、换行符
                if not line:  # 跳过空行
                    continue
                data_line = line.split(', ') # 分割后得到字段列表
                if len(data_line) == 5: # 检验字段
                    title = data_line[0]
                    description = data_line[1]
                    due_time = data_line[2]
                    created_time = data_line[3]
                    completed = data_line[4]
                else:
                    print(f"警告：第{line_num}行数据格式错误（字段数量不符），已跳过该行")
                    continue
                task = Task(
                    title=title,
                    description=description,
                    due_date=due_time,
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
