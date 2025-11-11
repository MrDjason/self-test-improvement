# Day 19 面对对象进阶

# 可见性和属性装饰器
"""
Python 通过属性名前缀下划线标识访问可见性
__属性名（双下划线）：约定为 “私有属性” ——
类外直接访问会报AttributeError（如stu.__name），但类内可通过self.__属性名访问
类外仍能通过实例._类名__属性名（如stu._Student__name）间接访问，并非绝对私有。

_属性名（单下划线）：约定为 “受保护属性”——仅作为提示告知开发者 “建议不直接外部访问”
"""
class Student:

    def __init__(self, name, age):
        self.__name = name
        self.__age = age

    def study(self, course_name):
        print(f'{self.__name}正在学习{course_name}.')


stu = Student('王大锤', 20)
stu.study('Python程序设计')
print(stu._Student__name) # 并无严格限定
try: 
    print(stu.__name)  # AttributeError: 'Student' object has no attribute '__name'
except AttributeError as e:
    print(f'遇到错误:{e}')

# 动态属性
# Python属于动态语言，运行时可给对象新增属性、方法，或删除已有结构
class Student:
    # __slots__ = ('name', 'age')
    # 在类中指定__slots__ = ('name', 'age')，这样Student类的对象只能有name和age属性
    def __init__(self, name, age):
        self.name = name
        self.age = age

stu = Student('王大锤', 20)
stu.sex = '男'  # 给学生对象动态添加sex属性
# 删除动态添加的sex属性
del stu.sex
try:
    print(stu.sex)  # 报错：AttributeError（属性已被删除）
except AttributeError as e:
    print(f'错误原因:{e}')

# 静态方法和类方法
class Triangle(object):
    """三角形"""

    def __init__(self, a, b, c):
        """初始化方法"""
        self.a = a
        self.b = b
        self.c = c
    # Triangle.is_valid(3,4,5) 时，Python 会直接找到 Triangle 类中用 @staticmethod 装饰的 is_valid 方法，判断逻辑
    @staticmethod
    def is_valid(a, b, c):
        """判断三条边长能否构成三角形(静态方法)"""
        return a + b > c and b + c > a and a + c > b
    # 声明类方法，可以使用classmethod装饰器
    """
    @classmethod
    def is_valid(cls, a, b, c):
        return a + b > c and b + c > a and a + c > b
    """
    # 二者的区别在于，类方法的第一个参数是类对象本身，而静态方法则没有这个参数
    def perimeter(self):
        """计算周长"""
        return self.a + self.b + self.c

    def area(self):
        """计算面积"""
        p = self.perimeter() / 2
        return (p * (p - self.a) * (p - self.b) * (p - self.c)) ** 0.5


# 1. 对象方法
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    # 对象方法：操作实例属性
    def introduction(self):
        print(f'我叫{self.name},今年{self.age}岁了。')

p = Person('John', 35)
p.introduction()

# 2. 类方法
class Person:
    count = 0 # 类属性：统计创建了多少个实例
    def __init__(self, name):
        self.name = name
        Person.count += 1 # 每次创建实例，类属性+1
    @classmethod
    def get_total(cls):
        # cls 就是 Person 类对象，cls.count 等价于 Person.count
        return f"总共创建了{cls.count}个人"
print(Person.get_total())  # 输出：总共创建了0个人

# 创建实例
p1 = Person("张三")
p2 = Person("李四")

print(p1.get_total())  # 输出：总共创建了2个人

# 3. 静态方法
class Person:
    # 静态方法：独立的工具函数，不依赖实例/类属性
    @staticmethod
    def is_adult(age):
        # 只处理传入的参数age，不涉及self或cls
        return age >= 18

# 使用：直接通过类调用
print(Person.is_adult(20))  # 输出：True
print(Person.is_adult(15))  # 输出：False

# 也可以通过实例调用
p = Person()
print(p.is_adult(18))  # 输出：True

# 继承和多态

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def eat(self):
        print(f'{self.name}正在吃饭')

    def sleep(self):
        print(f'{self.name}正在睡觉')

class Student(Person):
    def __init__(self, name, age):
        super().__init__(name, age) # super().__init__(name, age)  # 调用父类Person的初始化，给name、age赋值
    
    def study(self, course_name):
        print(f'{self.name}正在上{course_name}')

class Teacher(Person):
    def __init__(self, name, age, title):
        super().__init__(name, age)
        self.title = title
    def teach(self, course_name):
        print(f'{self.name}正在教{course_name}')
    
stu1 = Student('王大锤', 25)
stu2 = Student('李擂', 35)
stu1.sleep()
stu2.eat()
tec1 = Teacher('武忠祥', 55, '教授')
tec1.teach('高等数学')
stu1.study('高等数学')

# 父类：提供公共属性和方法的类
# 子类：继承父类并扩展功能的类

# 语法
'''
class 子类名(父类名):
    # 子类内容
若未指定父类,默认继承object(Python 所有类的顶级父类)
'''
# 子类对象可以 “替换” 父类对象使用