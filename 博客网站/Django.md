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
    '/var/www/static', # Linux地址
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

到这里我们就基本实现了博客的简单登录功能，这里实际项目的登录功能还有很大的距离，实际项目里我们要让他既支持用户名登录，又支持邮箱登录，再一个我们这个邮箱登录界面太过丑陋，下一章我们就来解决这些问题，进一步实现利用django的表单模块来完善我们登录功能。

### 在Django中构建表单

**通过forms.py自动生成表单**

上一节我们已经知道如何使用HTML定义表单，并获取表单的数据。那么这一节我们来学习如何利用django提供的表单类来构建美单，并进行有效性验证。

- **1.users下创建forms.py文件**
```python
from django import forms

class LoginForm(forms.Form):
    username = forms.CharField(label='用户名', max_length=15)
    password = forms.CharField(label='密码', min_length=6, widget=forms.PasswordInput())
```
这里定义了一个登陆框表单类，定义表单的字段类似于模型中的的字段，字段中的label标签实际生成就是表单的label标签，widget是定义表单的小部件，CharField字段的默认小部件是TextInput，意思也就是表单的类型是text文本类型，那么我们将密码这里的小部件定义为PasswordInput，它的表单类型就变成了password类。

- **2.修改views.py中逻辑**

```python
from django.shortcuts import render, HttpResponse
from django.contrib.auth import authenticate, login
from .forms import LoginForm

def login_view(request):
    if request.method != 'POST':
        form = LoginForm()
    else:
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return HttpResponse('登陆成功')
            else:
                return HttpResponse('账号或密码错误！')
    return render(request, 'users/login.html', {'form':form})
```

这个视图的逻辑是如果不是POST请求则显示空表单，否则获取表单数据，并进行验证数据类型是否正确，在进行验证登录操作，这里的数据都存储在cleaned_data这个字典中，通过获取字典数据的方式可以获取数据。

- **3.修改login.html中的代码**

```html
{{ form }}
```

- `{{ form.as_table }}`：将它们渲染为包裹在`<tr>`标签中的表格单元格
- `{{ form.as_p }}`：将它们包裹在`<p>`标签中
- `{{ form.as_ul }}`：将它们包裹在`<li>`标签中

- **4.自定义验证表单**
- 验证用户名与密码不能一致

```python
# forms.py class LoginForm(forms.Form):
def clean_password(self)
    username = self.cleaned_data.get('username')
    password = self.cleaned_data.get('password')
    if username == password:
        raise forms.validationError('密码不能与用户名一样！')
    return password
```

### 引入css样式

引入bulma.css样式，美化登录页面  

[bulma](https://bulma.io/)  

users下创建static文件夹，在这个文件夹下创建一个与项目同名的文件夹users

## 第五章 实现用户注册功能

> 首先大家来想一下我们以往看到别人的网站，是个什么样的流程？一般流程都是输入邮箱，用户名，密码，然后给邮箱发个激活链接，点击链接验证之后，这就算注册成功了。当然，现在的注册已经简单了，直接手机获取验证码，验证码填写正确后，在设置密码、用户名，那么我们这一章还是主要学习如何通过用户名、密码来注册，之后再来完善验证邮箱，通过邮箱完成注册逻辑。

**1. 在 users/views.py 中定义一个注册的视图**
```python
# users/views.py
def register(request):
    if request.method != 'POST':
        return HttpResponse('注册页面')
    else:
        return HttpResponse('POST提交页面')
```

**2. 在 users/urls.py 中定义注册页面的 url**
```python
path('register/', views.register, name='register'),
```

**3. 在 users/forms.py 中定义注册表单**
> 大家要清楚一点就是，我们要注册其实就是要在数据库创建一个用户，那么我们在前边就已经知道，django 为我们提供了一个User模型，里边有用户基本的字段，那么这个表单就与我们这个User模型相关，这种情况下，此时再在表单中定义字段类型就是多余的，因为在User模型中已经定义了字段，因此，Django 提供了一个辅助类ModelForm，可以从一个 User 模型创建一个 Form 类。

```python
from django.contrib.auth.models import User

class RegisterForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('username', 'password')
```


**4. 在users/templates/users/下创建注册页静态模板register.html**

**5. 修改 users/views.py 中的注册的视图如下**

下边会用到一个save()方法用来保存，其中save()方法的commit=False属性，是将验证过的数据暂时生成一个数据对象，暂存到内存，不会直接存入数据库，等用set_password()方法将明文密码转换为哈希值后，最后再存进数据库。

```python
from .forms import LoginForm, RegisterForm

def register(request):
    if request.method != 'POST':
        form = RegisterForm()
    else:
        form = RegisterForm(request.POST)
        if form.is_valid():
            new_user = form.save(commit=False)
            new_user.set_password(form.cleaned_data['password'])
            new_user.save()
            return redirect('users:login')
    context = {'form': form}
    return render(request, 'users/register.html', context)
```

**6. 将 users/forms.py 中的密码框设置为隐藏，并让在注册时输入两次密码，进行比对验证。**

```python
class RegisterForm(forms.ModelForm):
    password = forms.CharField(label='密码', min_length=6,
widget=forms.PasswordInput(attrs={
    'class': 'input', 'placeholder': '密码'}))
    password1 = forms.CharField(label='再次密码', min_length=6,
widget=forms.PasswordInput(attrs={
    'class': 'input', 'placeholder': '再次密码'}))

    class Meta:
        model = User
        fields = ('username', 'password')

    def clean_username(self):
        username = self.cleaned_data.get('username')
        exists = User.objects.filter(username=username).exists()
        if exists:
            raise forms.ValidationError("用户名已经存在！")
        return username

    def clean_password1(self):
        data = self.cleaned_data
        password = data['password']
        password1 = data['password1']
        if password != password1:
            raise forms.ValidationError('两次输入的密码不一致，请修改！')
        return password
```

### 实现邮箱登录功能

上一章节我们已经实现了利用用户名注册登录功能，这一章我们来学习如何使用邮箱注册。

**1. 在 users/views.py 中引入以下两个方法**

```python
from django.contrib.auth.backends import ModelBackend
from django.db.models import Q
```

ModelBackend是Django使用的默认身份验证后端，我们将重写这个类让他支持验证邮箱，它实现了两个必要的方法：`get_user(user_id)` 和 `authenticate(request, **credentials)`，以及其它一系列可选的权限相关的方法，具体我们可以去看官方文档或者直接查看源码。

Q方法可参考我的头条文章：如何在Django ORM中进行 or、and、not查询？  
`Q` 对象能通过 `&` 和 `|` 操作符连接起来。当操作符被用于两个 Q 对象之间时会生成一个新的 `Q` 对象。

**2. 在 users/views.py 中创建MyBackend类**

```python
class MyBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None):
        try:
            user = User.objects.get(Q(username=username)|Q(email=username))
            if user.check_password(password):  # 加密明文密码
                return user
        except Exception as e:
            return None
```

**3. 在MyBlog/settings.py中引入AUTHENTICATION_BACKENDS**

```python
AUTHENTICATION_BACKENDS = (
    'users.views.MyBackend',
)
```

经过这几步简单的配置我们就实现了使用邮箱登录功能。

### 实现使用邮箱注册功能

**1. 修改users/views.py中的注册视图register，在保存过程中设置用户名等于邮箱即可**

```python
def register(request):
    """注册视图"""
    if request.method != 'POST':
        form = RegisterForm()
    else:
        form = RegisterForm(request.POST)
        if form.is_valid():
            new_user = form.save(commit=False)
            new_user.set_password(form.cleaned_data.get('password'))
            # 让username等于邮箱即可
            new_user.username = form.cleaned_data.get('email')
            new_user.save()
            return HttpResponse('注册成功')
    context = {'form': form}
    return render(request, 'users/register.html', context)
```

**2. 修改users/forms.py中的RegisterForm表单如下**

```python
class RegisterForm(forms.ModelForm):
    '''注册表单'''
    email = forms.EmailField(label='邮箱', min_length=3, widget=forms.EmailInput(attrs={
        'class': 'input', 'placeholder': '邮箱'
    }))
    password = forms.CharField(label='密码', min_length=6, widget=forms.PasswordInput(attrs={
        'class': 'input', 'placeholder': '密码'}))
    password1 = forms.CharField(label='再次输入密码', min_length=6, widget=forms.PasswordInput(attrs={
        'class': 'input', 'placeholder': '再次输入密码'})

    class Meta:
        model = User
        fields = ('email', 'password')

    def clean_email(self):
        ''' 验证用户是否存在 '''
        email = self.cleaned_data.get('email')
        exists = User.objects.filter(email=email).exists()
        if exists:
            raise forms.ValidationError('邮箱已经存在！')
        return email

    def clean_password1(self):
        '''验证密码是否相等'''
        if self.cleaned_data['password'] != self.cleaned_data   ['password1']:
            raise forms.ValidationError('两次密码输入不一致！')
        return self.cleaned_data['password1']
```

这里我们将username表单变为email表单字段，将验证用户名是否存在修改为验证邮箱是否存在。

### 实现注册发送激活链接到邮箱，激活用户
上一章节我们已经实现了利用邮箱注册登录功能，那么这一章就正式学习如何向邮箱发送激活邮件，并激活用户。

**1. 在users/models.py中创建验证码字段**

```python
from django.db import models
class EmailVerifyRecord(models.Model):
    '''邮箱验证记录'''
    SEND_TYPE_CHOICES = (
        ('register', '注册'),
        ('forget', '找回密码'),
    )

    code = models.CharField('验证码', max_length=20)
    email = models.EmailField('邮箱', max_length=50)
    send_type = models.CharField(choices=SEND_TYPE_CHOICES, max_length=10, default='register')
    send_time = models.DateTimeField('时间', auto_now_add=True)

    class Meta:
        verbose_name = '邮箱验证码'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.code
```

- 激活虚拟环境，终端运行 makemigrations 生成数据迁移表
```bash
python manage.py makemigrations
```

- 运行migrate命令写入数据库
```python

```

**2.**

**3. 在 users 的同级目录创建一个 utils 文件夹，再在其内部创建一个 __init__.py 文件**

```
# 目录结构
users/
utils/__init__.py
```

**4. 在utils目录下创建一个email_send.py的文件，实现生成验证码的逻辑方法**

```python
from users.models import EmailVerifyRecord
from django.core.mail import send_mail
import random
import string
```

首先我们在 `email_send.py` 中引入了以上三个方法:
- `EmailVerifyRecord`是我们创建邮箱验证记录的类。
- `send_mail`是django默认为我们封装的邮件发送方法，这个封装方法可以加快邮件的发送，也可以在开发时测试发送邮件，在不支持SMTP的平台上支持发送邮件。
- random是python提供的模块，生成随机字符串
- string是python提供的字符串模块，这里用来生成字符串

```python
# 这是send_mail方法的一些参数
send_mail(subject, message, from_email, recipient_list,
          fail_silently=False, auth_user=None, auth_password=None,
          connection=None, html_message=None):
```

在大多数情况下，我们都可以使用 `django.core.mail.send_mail()` 来发送邮件。
其中参数 `subject`, `message`, `from_email` 和 `recipient_list` 是必须的。
- `subject`: 一个字符串。
- `message`: 一个字符串。
- `from_email`: 一个字符串。
- `recipient_list`: 一个字符串列表，每项都是一个邮箱地址。
- `fail_silently`: 一个布尔值。若为 `False`，`send_mail()` 会在发生错误时抛 `smtplib.SMTPException`。
- `auth_user`: 可选的用户名，用于验证登陆 SMTP 服务器。若未提供，Django 会使用 `EMAIL_HOST_USER` 指定的值。
- `auth_password`: 可选的密码，用于验证登陆 SMTP 服务器。若未提供，Django 会使用 `EMAIL_HOST_PASSWORD` 指定的值。

- `connection`: 可选参数，发送邮件使用的后端。若未指定，则使用默认的后端。查询 邮件后端 文档获取更多细节。

- `html_message`: 若提供了 `html_message`，会使邮件成为 `multipart/alternative` 的实例，`message` 的内容类型则是 `text/plain`，并且 `html_message` 的内容类型是 `text/html`。

返回值会是成功发送的信息的数量（只能是 0 或 1，因为同时只能发送一条消息）。


**7. 创建发送邮件后端，在MyBlog/settings.py中添加如下参数**

```python
EMAIL_HOST = 'smtp.163.com'      # 用于发送电子邮件的主机。
EMAIL_HOST_USER = "admin@163.com" # 自己的邮箱地址
EMAIL_HOST_PASSWORD = "123456"    # 自己的邮箱授权登录码
EMAIL_HOST_PORT = 465             # 用于中定义的SMTP服务器的端口
EMAIL_USE_SSL = True              # 是否使用隐式的安全连接
```

**8. 在users/views.py中查询验证码方法**
如果能根据验证码查询到用户，那么就激活这个用户，允许登录到后台

```python
def active_user(request, active_code):
    """查询验证码"""
    print(active_code)
    all_records = EmailVerifyRecord.objects.filter(code=active_code)
    print(all_records)
    if all_records:
        for record in all_records:
            email = record.email
            print(email)
            user = User.objects.get(email=email)
            print(user)
            user.is_staff = True
            user.save()
    else:
        return HttpResponse('链接有误！')
    return redirect('users:login')
```

### 忘记密码，通过注册邮箱发送邮件方式找回密码
**找回密码的逻辑：**
- **1. 登录页面点击"忘记密码"**
- **2. 上一步后，跳转到找回密码表单页面**
    - 表单提示填写登录邮箱
    - 判断邮箱是否注册，已注册，发送邮件；未注册，提示未注册消息，立即注册
- **3. 点击邮箱链接，创建邮箱链接视图**
    - 根据校验码获取邮箱地址
    - 根据邮箱获取用户
    - 修改用户密码
    - 修改成功后提示成功消息，跳转到登录页面

**详细步骤：**

1. 在users/templates/users/login.html登录模板中添加忘记密码跳转链接

```html
<a class=" is-pulled-right is-size-7" href="{% url 'users:forget_pwd' %}">忘记密码？</a>
```

2. 在users/forms.py文件中创建找回密码的表单类，提示其输入注册邮箱

```python
class ForgetPwdForm(forms.Form):
    """填写邮箱地址表单"""
    email = forms.EmailField(label='请输入注册邮箱地址', min_length=4,
widget=forms.EmailInput(attrs={
        'class': 'input', 'placeholder': '用户名/邮箱'
    }))
```

3. 在users/views.py视图文件中创建找回密码视图

```python
def forget_pwd(request):
    """找回密码"""
    if request.method == 'GET':
        form = ForgetPwdForm()
    elif request.method == 'POST':
        form = ForgetPwdForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data.get('email')
            exists = User.objects.filter(email=email).exists()
            if exists:
                # 发送邮件
                send_register_email(email, 'forget')
                return HttpResponse('邮件已经发送请查收！')
            else:
                return HttpResponse('邮箱还未注册，请前往注册！')
    return render(request, 'users/forget_pwd.html', {'form': form})
```

这里的逻辑也非常简单，首先判断是否为get请求，如果是，就显示空表单。在判断是否为POST请求，如果是显示表单并获取表单数据，验证数据的合法性之后，获取到用户输入表单的email地址跟数据库比对，看是否存在，如果存在发送一封邮件到邮箱，不存在则提示信息，并跳转到注册页面。这里的发送邮件我们沿用了注册发送邮件的方法，但这里判断了一个类型是否为forget，所以我们还得去utils/email_send.py中判断这个类型。

4. 在utils/email_send.py文件中判断forget类型发送邮件的内容  
> 在send_register_email方法下判断完注册类型发送邮件之后紧接着判断找回密码类型，与注册是的写法一样。

```python
elif send_type == 'forget':
    email_title = '轻编程博客密码重置链接'
    email_body = '请点击下面链接激活你的账号：
http://127.0.0.1:8000/users/forget_user/{0}'.format(code)

    send_status = send_mail(email_title, email_body, '15336112587@163.com', [email])
    if send_status:
        pass
```

5. 在users/urls.py中创建找回密码表单页面url

```python
path('forget_pwd/', views.forget_pwd, name='forget_pwd'),  # 找回密码发送邮件页面
```
测试是否能成功发送邮件。

## 第六章 实现验证邮箱激活账户功能

在一个网站中，既然存在登录，注册功能，那么必然会存在用户中心，我们的博客不允许每个人都发表文章，但允许每个用户浏览文章，收藏文章，并且可以点赞，留言，那么用户中心就用来体现他自己的这些操作，并修改自己的相关信息。

1. 在users\views.py中创建用户中心视图

```python
from .models import UserProfile  # 引入用户模型字段
from django.contrib.auth.decorators import login_required

@login_required(login_url='users:login')  # 设置登录后才能访问，如果没有登陆，就跳转到登录界面
def user_profile(request):
    user = User.objects.get(username=request.user)
    return render(request, 'users/user_profile.html', {'user': user})
```

2. 创建个人中心的静态模板 users/templates/users/user_profile.html
```python
# 
```

3. 在users/urls.py文件中定义个人中心url

```python
path('user_profile/', views.user_profile, name='user_profile')
```




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