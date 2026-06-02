# Django

## 第一章 课程介绍

## MVC与MTV是什么？
两者都是项目目录结构的设计模式，无论是采用MVC模式还是MTV模式，其本质都是一样的，它的核心目的就是为了解耦代码，提高开发效率。类似于我们前端将HTML、JS及css分离到不同的文件，而不是集中在一个HTML文件里的思路是一致的，便于维护、便于开发。

- **MVC（Model View Controller 模型-视图-控制器）**
  - M：model，模型，就是数据模型，负责数据的存取；
  - V：view，视图（模板），主要负责页面的展示逻辑；
  - C：controller，控制器（视图函数），负责业务逻辑的处理；（对应django的V）

- **MTV（Model Templates View 模型-模板-视图）**
  - M：model，模型，就是数据模型，负责数据的存取；
  - T：template，模板，负责页面的展示逻辑；（对应MVC的V，用户真正看到的东西）
  - V：view function，视图函数，负责业务逻辑的处理；

## 第二章 安装Django创建项目

- 创建虚拟环境

```python
python -m venv env
```

- 激活虚拟环境

```python
env\Scripts\activate
```

- 创建项目

```python
django-admin startproject MyBlog
```

- 熟悉项目结构

```bash
 ├── 项目名
 │   ├── asgi.py # 与ASGI兼容的Web服务器为项目提供服务的入口
 │   ├── __init__.py # 让这个目录被认为是一个Python包
 │   ├── settings.py # Django项目配置文件
 │   ├── urls.py # URL声明，相当于网站目录
 │   └── wsgi.py # 作为项目运行在WSGI兼容的Web服务器上的入口
 └── manage.py # 管理Django项目的命令行工具
```

- 创建子应用

```python
py manage.py startapp blog
```

- 启动项目

```python
py manage.py runserver
```

## 第三章 创建用户中心应用

- 配置语言时区

```python
# settings.py

LANGUAGE_CODE = 'zh-hans'
TIME_ZONE = 'Asia\Shanghai'
```

- 配置静态文件的目录列表和上传url

```python
# settings.py
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
    '/var/www/static',
]

# urls.py
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
```

- 创建用户中心应用

```python
py manage.py startapp users
```

在创建好`users`应用之后，在`users`文件夹下的`models.py`中创建`UserProfile`模型数据。  
Django的模型是一个个类，模型数据在应用`users`中`models.py`中定义，直接与数据库交互。  
在创建模型数据之前，首先把应用注册到`settings.py`文件中  
注册方法为在项目设置文件`settings.py`中的INSTALLED_APPS设置中添加包含models.py文件模块的名字。
**注册方式有两种**：

```python
INSTALLED_APPS = [
    'users', # 第一种，Django 会自动寻找 users 应用下的 apps.py 文件，找到默认的配置类（通常是 UsersConfig）
    'users.apps.UsersConfig', # 第二种，明确指定使用 users/apps.py定义的UsersConfig
]

```

## 第四章 实现用户登录功能

## 第五章 实现用户注册功能

## 第六章 实现验证邮箱激活账户功能

## 第七章 实现用户中心

## 第八章 创建博客应用

## 第九章 完善博客模型

## 第十章 实现博客文章列表页

## 第十一章 实现博客详情页

## 第十二章 实现发布文章识别Markdown语法

## 第十三章 实现侧边栏功能

## 第十四章 实现用户收藏文章功能

## 第十五章 实现全局搜索功能

## 第十六章 实现文章评论功能

## 第十七章 集成ckeditor5富文本编辑器

## 第十八章 实现阅读量统计功能

## 第十九章 部署上线