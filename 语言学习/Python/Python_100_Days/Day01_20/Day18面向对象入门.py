# Day 18 面向对象入门

# 序中的数据和操作数据的函数是一个逻辑上的整体，我们称之为对象
# 对象可以接收消息，解决问题的方法就是创建对象并向对象发出各种各样的消息

# 面向对象编程：把一组数据和处理数据的方法组成对象，把行为相同的对象归纳为类
# 通过封装隐藏对象的内部细节，通过继承实现类的特化和泛化，通过多态实现基于对象类型的动态分派。

# 对象、类、封装、继承、多态
# 类是对象的蓝图和模板，对象是类的实例
# 定义一个"狗"类（模板）
"""
class Dog:
    # 属性：所有狗都有品种、年龄、颜色
    def __init__(self, breed, age, color):
        self.breed = breed
        self.age = age
        self.color = color
    
    # 方法：所有狗都能吠叫
    def bark(self):
        print("汪汪！")
"""
# 封装指将对象的属性和方法捆绑在一起，并隐藏内部实现细节，只通过公开的 “接口”（方法）与外部交互。
"""
class Dog:
    def __init__(self, age):
        self.__age = age  # 用双下划线标记为"私有属性"，外部不能直接访问
    
    # 公开接口：获取年龄
    def get_age(self):
        return self.__age
    
    # 公开接口：修改年龄（带校验逻辑）
    def set_age(self, new_age):
        if new_age > 0:  # 确保年龄合法
            self.__age = new_age
        else:
            print("年龄不能为负数！")
"""
# 继承指一个类（子类）可以继承另一个类（父类）的属性和方法，同时子类可以扩展新功能或修改父类的功能。
"""
# 父类：动物
class Animal:
    def breathe(self):
        print("呼吸空气")
    
    def eat(self):
        print("吃东西")

# 子类：狗（继承自动物）
class Dog(Animal):
    # 继承父类的breathe()和eat()，无需重写
    # 新增子类特有方法
    def bark(self):
        print("汪汪！")

# 子类：猫（继承自动物）
class Cat(Animal):
    # 继承父类的方法
    # 新增子类特有方法
    def meow(self):
        print("喵喵！")

# 使用：子类对象可以调用父类的方法
wangcai = Dog()
wangcai.breathe()  # 输出"呼吸空气"（继承自Animal）
wangcai.bark()     # 输出"汪汪！"（子类特有）
"""
# 多态指不同类的对象对同一操作（方法调用）做出不同响应。简单说：“同一接口，不同实现”。
"""
class Animals:
    def make_sound(self):
        pass
    
class Dog(Animals):
    def make_sound(self):
        print("wangwang")

class Cat(Animals):
    def make_sound(self):
        print('miaomiao')
    
# 统一调用接口，无需关心具体类型
def animal_sound(animal):
    animal.make_sound() # 多态：传入Dog则汪汪，传入Cat则喵喵

xiaohuang = Dog()
animal_sound(xiaohuang)


def dog_bark(dog):   # 参数 dog 用来接收一个对象
    dog.make_sound() # 调用对象的方法

dog_bark(Dog()) # 处理对象，Dog() 是 创建对象的操作
"""
# 定义类
class Student:
    def study(self, course_name):
        print(f'学生正在学习{course_name}')
    def play(self):
        print(f'学生正在玩游戏')

# 创建对象
stu1 = Student()
stu2 = Student()
print(stu1)    # <__main__.Student object at 0x10ad5ac50>
print(stu2)    # <__main__.Student object at 0x10ad5acd0> 
print(hex(id(stu1)), hex(id(stu2)))    # 0x10ad5ac50 0x10ad5acd0 hex()函数将整数转化为十六进制字符串
# 打印对象变量（如 print (stu1)）会显示对象内存地址（十六进制）
# 与 id () 函数查得的对象标识一致，变量实际保存的是对象的逻辑地址，通过该地址可找到对象

# 方法第一个参数 self 代表接收消息的对象，study方法还有课程名称参数，Python调用对象方法有两种方式——
Student.study(stu1, 'Python程序设计')
stu1.study('Python程序设计')

Student.play(stu1)
stu1.play()

# 初始化方法__init__
# 通过给Student类添加__init__方法的方式为学生对象指定属性，同时完成对属性赋初始值的操作
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def study(self, course_name):
        print(f'{self.name}正在学{course_name}')

    def play(self):
        print(f'{self.name}正在玩')

stu1 = Student('王大锤', 25)
stu1.study('Python程序设计')
stu1.play()

import time


# 定义时钟类
class Clock:
    """数字时钟"""

    def __init__(self, hour=0, minute=0, second=0):
        """初始化方法
        :param hour: 时
        :param minute: 分
        :param second: 秒
        """
        self.hour = hour
        self.min = minute
        self.sec = second

    def run(self):
        """走字"""
        self.sec += 1
        if self.sec == 60:
            self.sec = 0
            self.min += 1
            if self.min == 60:
                self.min = 0
                self.hour += 1
                if self.hour == 24:
                    self.hour = 0

    def show(self):
        """显示时间"""
        return f'{self.hour:0>2d}:{self.min:0>2d}:{self.sec:0>2d}'
        # {self.hour:0>2d} 里的 :0>2d 是格式说明符
        # 0 当 self.hour 的位数不足指定宽度时，用 0 填充空缺位置
        # > 表示右对齐（< 是左对齐，^ 是居中对齐），即数字靠右，填充字符靠左。
        # 2 定义格式化后字符串的固定长度为 2
        # d 表示待格式化的值是整数

# 创建时钟对象
clock = Clock(23, 59, 58)
start_time = time.time()  # 记录程序开始时间（秒）
run_duration = 10  # 设定程序运行10秒后自动结束

# 有条件循环
while time.time() - start_time < run_duration:
    print(clock.show(), end='\r')
    time.sleep(1)
    clock.run()

# 平面上的点
class Point:
    """平面上的点"""

    def __init__(self, x=0, y=0):
        """初始化方法
        :param x: 横坐标
        :param y: 纵坐标
        """
        self.x, self.y = x, y

    def distance_to(self, other):
        """计算与另一个点的距离
        :param other: 另一个点
        """
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx * dx + dy * dy) ** 0.5

    def __str__(self):
        return f'({self.x}, {self.y})'


p1 = Point(3, 5)
p2 = Point(6, 9)
print(p1)  # 调用对象的__str__魔法方法
print(p2)
print(p1.distance_to(p2))