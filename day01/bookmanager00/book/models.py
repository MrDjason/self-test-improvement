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

'''
1.模型类 需要继承自 models.Model
2.定义属性
    属性名=models.类型(选项)
    2.1属性名 对应 就是字段名
       不要是使用python、Mysql关键字、连续__
    2.2类型MySQL的类型
    2.3选项 是否有默认值，是否唯一，是否为null
       CharField必须设置max_length
'''
class BookInfo(models.Model):
    # 创建字段，字段类型...
    # id已创建
    name = models.CharField(max_length=10, unique=True)
    pub_date = models.DateField(null=True)
    readcount = models.IntegerField(default=0)
    commentcount = models.IntegerField(default=0)
    is_delete = models.BooleanField(default=False)

    def __str__(self):
        return self.name

    class Meta:
    # Meta是模型的 “配置项”，专门用来定制模型的非字段相关规则。
        db_table = 'bookinfo' # 修改表的名字
        verbose_name = '书籍管理' # admin站点使用

class PeopleInfo(models.Model):
    # 定义一个有序字典
    GENDER_CHOICE = (
        (1, 'male'),
        (2, 'female')
    )
    name = models.CharField(max_length=10,unique=True)
    gender = models.SmallIntegerField(choices=GENDER_CHOICE, default=1)
    description = models.CharField(max_length=100,null=True)
    is_delete=models.BooleanField(default=False)
    # 外键约束：人物属于哪本书
    # 系统会自动给为外键添加_id

    # 外键级联操作
    book = models.ForeignKey(BookInfo,on_delete=models.CASCADE)
    class Meta:
        db_table='peopleinfo'