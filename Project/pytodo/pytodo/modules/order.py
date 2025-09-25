class Task:
    def __init__(self, title, description, due_time, created_time, completed = False):
        """
        :param titie        : 标题(10个字符)
        :param description  : 描述(30个字符)
        :param due_time     : 预计完成时间,格式为:YYYY-MM-DD-HH-mm
        :param created_time : 创建时间，格式为YYYY-MM-DD
        :param completed    : 完成状态(布尔值,默认 False)
        """
        self.title = title
        self.description = description
        self.due_time = due_time
        self.created_time = created_time
        self.completed = completed
    
    def __str__(self):
        # 重写字符串方法，便于打印任务信息
        return (f'标题:{self.title:<10} 描述:{self.description:<30} \n'
            f'预计完成时间:{self.due_time} 创建时间:{self.created_time} 完成状态:{self.completed}')