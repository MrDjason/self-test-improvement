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
TIME_ZONE = 'Asia/Shanghai'
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

注册完成后，开始在users文件夹下的`models.py`添加数据

```python
# 首先引入django内置的一个用户User模型，然后通过一对一关联关系为默认的User扩展用户数据
from django.contrib.auth.models import User
```

User是继承了AbstractUser类的所有数据，该类里面创建了用户的几个基本信息，如`username`、`first_name`、`last_name`、`email`、`is_staff`、`is_active`、`date_joined`  

这几个字段不能完全概括，于是需要扩展用户模型。  

扩展用户模型有两种方法，一种是模仿官方的方法，直接继承`AbstractUser`类来扩展用户数据，另一种是通过一对一关系关联User类来扩展用户数据。这里采用第二种方法，好处是不需要再项settings.py再配置  

```python
class UserProfile(models.Model):
    owner = models.OneToOneField(User,
    on_delete=models.CASCADE, # 删除则关联数据全部删除
    verbose_name = '用户' # 定义名称
    )
```

django提供了定义三种最常见的数据库关联关系的方法：$$多对一，多对多，一对一。$$  
我们默认的User扩展数据便是一对一关系`OneToOneField`
多对一关联为`ForeignKey`，多对多关联为`ManyTOManyField`

继续拓展用户信息

```python
class UserProfile(models.Model):
    USER_GENDER_TYPE = (
        ('male', '男'),
        ('female', '女')
    )
    owner = models.OneToOneField(User, on_delete=models.CASCADE, verbose_name = '用户')
    nike_name = models.CharField('昵称', max_length=50, blank=True, default='')
    birthday = models.DateField('生日', null=True, blank=True)
    gender = models.CharField('性别', max_length=6, choices=USER_GENDER_TYPE)
    address = models.CharField('地址', max_length=100, blank=True, default='')
    image = models.ImageField(upload_to='images/%Y/%m', default='images/default.png', max_length=100, verbose_name='用户头像')
```

- `CharField`：标题字段，用来设置用户的昵称
- `DateField`：时间字段，类似的还有DateTimeField
- `ImageField`：图像字段，`upload_to`用来指定用于上传文件的`MEDIA_ROOT`子目录，例如我们最开始配置`settings.py`的时候配置了`MEDIA_ROOT=os.path.join(BASE_DIR, 'media')`。这里的`upload_to='images/%Y/%m'`中upload_to的`'%Y / %m /%d'`部分是strftime()格式，`%Y`是四位数年份，`%m`是两位数月份，`%d`是两位数日期。如果在2020年1月15日上传文件，他将保存在目录/media/images/2020/01中

```python
# 在settings.py中TEMPLATES的OPTIONS添加
'django.template.context_processors.media',
```

使用`ImageField`字段还必须安装`pillow`模块，django默认使用pillow模块来处理图片  

`CharField`及其子类必须传入`max_length`参数，用于限定数据库`VARCHAR`类型存储占用的字节长度。

- 1. null
  - 取值：`True/False`，默认`False`
  - 作用：**数据库层级控制**
设为`True`时，字段为空则在数据库存入`NULL`；默认不允许数据库字段为空。

- 2. blank
  - 取值：`True/False`，默认`False`
  - 作用：**表单校验层级控制**
     - `blank=True`：表单提交校验时，该字段允许空值提交
     - `blank=False`：表单提交校验时，该字段必填、不可为空
  - 区分：`null`只管数据库，`blank`只管前端表单验证，二者生效层级不同。

- 3. choices
  - 1. 格式：由**二元组**组成的序列，用于字段枚举选项
  - 2. 表单效果：配置后表单控件自动变为下拉选择框，不再是普通输入框，输入内容被限定在枚举选项内
  - 3. 存储规则：二元组**第一项存入数据库**，**第二项仅前端展示**
  - 4. 获取展示值：模型实例调用 `get_字段名_display()` 方法，即可取出choices里的展示文本

### 生成默认的数据库sqlite3

Python内置了SQLite，通过一个命令快速生成sqlite数据库文件

```python
py manage.py migrate
```

开发阶段使用默认的数据库快速开发，真正部署的时候更换为mysql数据库  
然后通过运行makemigrations命令，来生成迁移文件，生成的迁移文件在migrations文件夹下

```python
py manage.py makemigrations users
```

再通过migrate命令  

```python
py manage.py migrate
```

这个命令将选中所有还没有执行过的迁移，将未迁移的数据同步到数据库中。改变模型只需要三步：

- 编辑`models.py`文件，改变模型
- 运行`python manage.py makemigrations`为模型的改变生成迁移文件
- 运行`python manage.py migrate`来应用数据库迁移

创建一个超级管理员用户

```python
py manage.py createsuperuser
```

向管理页面加入扩展的用户应用  

```python
admin.site.register(UserProfile)
```

打开[http://127.0.0.1:8000/admin](http://127.0.0.1:8000/admin)  
打开管理站点页面首页，发现有用户这一项，那么我们将关联的信息也关联到这一项里面，项users/admin.py管理页面注册UserProfile模型

```python
from django.contrib import admin

from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from .models import UserProfile # 引入定义的模型

# 必须先通过unregister将User取消注册
admin.site.unregister(User)

# 定义关联对象的样式，StackedInline为纵向排列每一行，TabularInline为并排排列
class UserProfileInline(admin.StackedInline):
    model = UserProfile # 关联的模型

# 关联字段在User之内编辑，关联进来
class UserProfileAdmin(UserAdmin):
    inlines = [UserProfileInline]

# 重新注册User
admin.site.register(User, UserProfileAdmin)
```

注册成功可以添加一个新用户进行测试。现在用户里面成功多出了我们添加的条目，但这些信息的**标题**显示的是模型的名称大写字母  
在`users/models.py`中的UserProfile模型下定义：

```python
from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    '''
    owner = models.OneToOneField(User,
    on_delete=models.CASCADE, # 删除则关联数据全部删除
    verbose_name = '用户' # 定义名称
    )
    '''
    USER_GENDER_TYPE = (
        ('male', '男'),
        ('female', '女')
    )
    owner = models.OneToOneField(User, on_delete=models.CASCADE, verbose_name = '用户')
    nike_name = models.CharField('昵称', max_length=50, blank=True, default='')
    birthday = models.DateField('生日', null=True, blank=True)
    gender = models.CharField('性别', max_length=6, choices=USER_GENDER_TYPE)
    address = models.CharField('地址', max_length=100, blank=True, default='')
    image = models.ImageField(upload_to='images/%Y/%m', default='images/default.png', max_length=100, verbose_name='用户头像')

    class Meta:
        verbose_name = '用户数据'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.owner.username
```
## 第四章 实现用户登录功能

> 学习宗旨：了解django的表单的一系列操作，包括获取表达数据，保存表单数据，以及了解django的视图、模板之间的关联关系以及工作机制。

要开发登录功能首先要有一个明确的开发思路：  
一、首先，要有一个前端可以看到登录页面，这里面牵扯到静态html页面以及一个url地址  
二、其次，要有一套处理登录逻辑的相关代码  
三、最后，定义登陆页面的url地址  

- 1.在应用下创建一个`templates`文件夹，再在这个文件夹下创建与应用同名的文件夹，以users应用为例，静态模板的路径为`users/templates/users`。Django能找到该目录下的文件，主要是因为`settings.py`中`TEMPLATES`中'DIRS':[os.path.join(BASE_DIR, 'templates)]。
- 2.登录功能需要输入用户名或者邮箱地址，再以密码登录，那么再users文件夹创建一个名为`login.html`的静态登录页面，并实现登录框
```html
<form action="" method="post">
    <label for="id_username">用户名:</label>
    <input type="text" name="username" id="id_username">
    <label for="id_passwprd">密码:</label>
    <input type="password" name="password" id="id_passord">
    <input type="submit" value="提交">
</form>
```
- 3.登录逻辑相关代码，在应用users下的视图views.py完成逻辑

```python
# users/views.py
from django.shortcuts import render

def login(request):
    return render(request, 'users/login.html')
```

- 4.定义登陆页面url地址  
在应用users下新建一个urls.py文件，引入如下代码

```python
from django.urls import path
from . import views

app_name = 'users'
urlpatterns = [
    path('login.html', views.login, name='login')
]
```

下一步是在根URLconf文件中指定我们创建的`users.urls`模块。在`mysite/urls.py`文件中的`urlpatterns`列表里插入一个`include()`

```python
from django.contrib import admin
from django.urls import include, path
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('users/', include('users.urls')), # 使用include()引入users.urls文件
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# 配置用户上传url
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

```
函数`include()`允许引用其它URLconfs，每当Django遇到`include()`时，它会截断与此项匹配的URL的部分，并将剩余的字符串发送到URLconf以供进一步处理。既可以实现即插即用的效果，可以放在你自己想要的任何路径下。  
[http://127.0.0.1:8000/users/login.html](http://127.0.0.1:8000/users/login.html)即可看到登录页面。  

### 获取表单数据

处理表单会用到GET和POST两种方法，GET 和 POST 通常用于不同的目的。  
01、任何可能用于更改系统状态的请求应该使用 POST —— 比如一个更改数据库的请求。GET 应该只被用于不会影响系统状态的请求。  
02、GET 方法也不适合密码表单，因为密码会出现在URL中，于是也会出现在浏览器的历史记录以及服务器的日志中，而且都是以纯文本的形式。它也不适合处理大量的数据或者二进制数据，比如一张图片。在WEB应用的管理表单中使用 GET 请求具有安全隐患：攻击者很容易通过模拟请求来访问系统的敏感数据。 POST 方法通过与其他像Django的CSRF token 这样的保护措施配合使用，能对访问提供更多的控制。  
另一方面，GET 方法适用于诸如网页搜索表单这样的内容，因为这类呈现为一个 GET 请求的URL很容易被存为书签、分享或重新提交。  
因此上，我们这里要实现登录功能，就必须使用POST方法。  
我们刚才在`users/views.py`中定义了一个login的登录函数，下来在这个函数中获取表单数据:

```python
# users/views.py
def login(request):
    if request.method == 'POST':    # 判断采用的是何种请求
        username = request.POST['username']    # request.POST[]或request.POST.get()获取数据
        password = request.POST['password']
        print(username, password)

    return render(request, 'users/login.html')
```

我们来解读以上代码: `if request.method`用来判断网页提交采用的是何种方法，也就是我们所说的GET方法和POST方法，当然还有其他方法，那我们最常用的就是这两种处理数据的方法，这里我们判断看是否为POST请求。

然后使用`django`为我们提供的`request.POST`方式获取表单数据，其中`request.POST`后边下标中传入的是表单的`name`属性的值。与`request.POST`方法类似的还设有`request.GET`以及`request.FILES`方法分别用来获取GET请求数据和二进制文件数据，使用方法与POST请求方式类似，也是非常简单，在后边具体用到的时候再说。

下来我们打开登录页面，随便输入一组数据，点击提交，发现以下403错误：
> 禁止访问 (403)
> CSRF验证失败.请求被中断.
>
> **Help**
> Reason given for failure:
> CSRF token missing or incorrect

提示信息为：**CSRF验证失败，请求被中断。为什么会出现这样的错误提示？**

这是因为，`Django`自带一个简单易用的跨站请求伪造防护。当通过 `POST` 方法提交一张启用了CSRF防护的表单时，您必须在静态模板中的表单前插入 `{% csrf_token %}` 模板标签。或者注销掉`settings.py`中`MIDDLEWARE`下的 `'django.middleware.csrf.CsrfViewMiddleware'` 这一行代码。这里我们采用第一种方法，更安全有效。

```html
<!-- users/templates/users/login.html 完整表单代码-->
<form action="" method="post">
    {% csrf_token %}
    <label for="id_username">用户名：</label>
    <input type="text" name="username" id="id_username">
    <label for="id_password">密码：</label>
    <input type="password" name="password" id="id_password">
    <input type="submit" value="提交">
</form>
```

### 实现登录逻辑
django默认提供了一个用户名和密码验证方法`authenticate`和登录方法`login`，首先在视图中引入，引入之后我们发现我们定义的`login`方法与django官方提供的名称一致，那么这里肯定会有问题，具体我不再演示，大家可以自己去尝试，我这里先把我定义的`login`方法改为`login_view`，对应的我们也要去修改一下`urls.py`中的方法。

```python
# users/views.py
from django.shortcuts import render, HttpResponse
from django.contrib.auth import authenticate, login

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        # 与数据库中的用户名和密码比对，django默认保存密码是以哈希形式存储，并不是明文密码，这里的password验证默认调用的是User类的check_password方法，以哈希值比较。
        user = authenticate(request, username=username, password=password)
        # 验证如果用户不为空
        if user is not None:
            # login方法登录
            login(request, user)
            # 返回登录成功信息
            return HttpResponse('登陆成功')
        else:
            # 返回登录失败信息
            return HttpResponse('登陆失败')

    return render(request, 'users/login.html')
```

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