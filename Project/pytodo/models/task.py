class Task:
    """
    任务数据结构类，存储任务核心属性
    :param title: 任务标题（字符串）
    :param description: 任务描述（字符串）
    :param due_date: 截止日期（格式：YYYY-MM-DD，字符串）
    :param due_time: 截止时间（格式：HH:MM，字符串）
    :param created_at: 创建时间（格式：YYYY-MM-DD，字符串）
    :param completed: 完成状态（布尔值，默认 False）
    """
    # 任务类定义

    def __init__(self, title, description, due_date, due_time, created_at, completed=False):
        self.title = title
        self.description = description
        self.due_date = due_date
        self.due_time = due_time  
        self.created_at = created_at
        self.completed = completed  # 任务完成状态，默认未完成

    def __str__(self):
        """重写字符串方法，便于打印任务信息"""
        return f"Task(title={self.title}, due_date={self.due_date}, due_time={self.due_time}, completed={self.completed})"
