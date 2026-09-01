# Django

## 1.激活虚拟环境

```bash
source venv/bin/activate
```

## 2.用django创建项目

```python
django-admin startproject 项目名
```

## 3.创建子应用

进入项目文件夹，文件夹内有manage.py

```python
python manage.py startapp 子应用名
```

## 4.修改配置

**项目setting设置**

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    '子应用1', # 新增
    '子应用2', # 新增
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql', # 修改前——'ENGINE': 'django.db.backends.sqlite3',
        'HOST': 'localhost', # 新增
        'PORT': 3306, # 新增
        'USER': 'root', # 新增
        'PASSWORD': 'password', # 新增
        'NAME': 'mysql内database名称', # 修改前——'NAME': BASE_DIR / 'db.sqlite3',
        # MYSQL创建新数据库——create database xxx charset utf8;
    }
}
```

## 5.运行项目

**命令台运行**

```python
python manage.py runserver
```

## 6.models文件编写

## 7，迁移

```python
# 生成迁移文件
python manage.py makemigrations
# 同步到数据库
python manage.py migrate
```

## 8.添加测试文件

插入数据到MySql中

```python
INSERT INTO bookinfo(name, pub_date, readcount, commentcount, is_delete)
VALUES
    ('射雕英雄传', '1980-5-1', 12, 34, 0),
    ('天龙八部', '1986-7-24', 36, 40, 0),
    ('笑傲江湖', '1995-12-24', 20, 80, 0),
    ('雪山飞狐', '1987-11-11', 58, 24, 0);

INSERT INTO peopleinfo(name, gender, book_id, description, is_delete)
VALUES
    ('郭靖', 1, 1, '降龙十八掌', 0),
    ('黄蓉', 0, 1, '打狗棍法', 0),
    ('黄药师', 1, 1, '弹指神通', 0),
    ('欧阳锋', 1, 1, '蛤蟆功', 0),
    ('梅超风', 0, 1, '九阴白骨爪', 0),
    ('乔峰', 1, 2, '降龙十八掌', 0),
    ('段誉', 1, 2, '六脉神剑', 0),
    ('虚竹', 1, 2, '天山六阳掌', 0),
    ('王语嫣', 0, 2, '神仙姐姐', 0),
    ('令狐冲', 1, 3, '独孤九剑', 0),
    ('任盈盈', 0, 3, '弹琴', 0),
    ('岳不群', 1, 3, '华山剑法', 0),
    ('东方不败', 0, 3, '葵花宝典', 0),
    ('胡斐', 1, 4, '胡家刀法', 0),
    ('苗若兰', 0, 4, '黄衣', 0),
    ('程灵素', 0, 4, '医术', 0),
    ('袁紫衣', 0, 4, '六合拳', 0);
```

## 9.子应用book/views定义函数

```python
from django.http import HttpResponse

def create_book(request):
    return HttpResponse('create')
```

## 10.子应用book/urls

在项目中添加urls.py  

`ctrl+.`可以搜索函数其他导入匹配项，自动添加import  

```python
# urls.py 定义子应用路由
from django.urls import path
from book.views import create_book

urlpatterns = [
    path('create/', create_book)
]
```

## 11.项目bookmanager01/urls.py

```python
from django.contrib import admin
from django.urls import path,include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('book.urls')),
]
```