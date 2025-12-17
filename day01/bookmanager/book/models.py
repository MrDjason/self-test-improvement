from django.db import models

# Create your models here.
'''
1.模型类需要继承自models.Model
    models.Model 会解析子类中定义的字段，将其转换为数据库表的列
    如果不继承 models.Model，其属性只是普通类属性，Django完全无法识别这是数据库列定义，更不会生成对应的表结构

2.系统会自动为我们添加一个主键id
3.字段
    字段名=model.类型（选项）
    字段名其实就是数据表的字段名
    字段名不要使用python、mysql等关键字
'''
class BookInfo(models.Model):
    # 创建字段，字段类型...
    # id已创建
    name = models.CharField(max_length=10)

class PeopleInfo(models.Model):
    name = models.CharField(max_length=10)
    gender = models.BooleanField()
    # 外键约束：人物属于哪本书
    book = models.ForeignKey(BookInfo,on_delete=models.CASCADE)
