# Django 完整教程 —— 从零到博客网站

> 本教程基于你已完成的博客网站项目，逐文件、逐行解释每个部分的作用和原理。

---

## 目录

1. [Django 是什么？MTV 架构](#第一章-django-是什么mtv-架构)
2. [项目结构详解](#第二章-项目结构详解)
3. [settings.py —— 项目配置核心](#第三章-settingspy--项目配置核心)
4. [models.py —— 数据库模型](#第四章-modelspy--数据库模型)
5. [数据库迁移 Migrations](#第五章-数据库迁移-migrations)
6. [forms.py —— 表单验证](#第六章-formspy--表单验证)
7. [views.py —— 视图函数（核心业务逻辑）](#第七章-viewspy--视图函数核心业务逻辑)
8. [urls.py —— URL 路由](#第八章-urlspy--url-路由)
9. [templates —— HTML 模板](#第九章-templates--html-模板)
10. [admin.py —— 后台管理](#第十章-adminpy--后台管理)
11. [邮件发送模块](#第十一章-邮件发送模块)
12. [完整的请求流程总结](#第十二章-完整的请求流程总结)

---

## 第一章 Django 是什么？MTV 架构

### 1.1 Django 是什么？

**Django** 是一个用 Python 写的 Web 框架。它的核心思想是：**帮你快速搭建网站，你只需要关心业务逻辑，不用重复造轮子。**

打个比方：如果盖房子，Django 已经给你准备好了地基、水管、电线管道，你只需要设计房间格局和装修即可。

### 1.2 MTV 架构

Django 采用 **MTV** 设计模式：

```
用户浏览器  →  请求某个URL  →  Django 服务器
                                   ↓
                              urls.py（路由表）
                              找到匹配的URL
                                   ↓
                              views.py（视图函数）
                              处理业务逻辑
                              从数据库拿数据
                                   ↓
                              models.py（数据模型）
                              与数据库交互
                                   ↓
                              templates（HTML模板）
                              渲染成HTML页面
                                   ↓
                              返回给用户浏览器
```

- **M（Model 模型）**：定义数据结构，和数据库打交道。对应你项目中的 `models.py`
- **T（Template 模板）**：HTML 页面，决定用户看到什么。对应你项目中的 `templates/` 文件夹
- **V（View 视图）**：核心逻辑，收到请求后决定做什么、返回什么。对应你项目中的 `views.py`

> 注意：Django 的 View 相当于 MVC 中的 Controller（控制器）。名字容易搞混，记住：**Django 的 V 是处理逻辑的，T 才是展示页面的。**

---

## 第二章 项目结构详解

你的项目结构如下：

```
MyBlog/                          ← 项目根目录（外层）
├── manage.py                    ← Django 命令行入口
├── db.sqlite3                   ← SQLite 数据库文件
├── MyBlog/                      ← 项目配置目录（内层，同名）
│   ├── __init__.py              ← 标识这是一个 Python 包
│   ├── settings.py              ← 全局配置文件
│   ├── urls.py                  ← 总路由表
│   └── wsgi.py                  ← 部署用的入口文件
├── blog/                        ← blog 子应用（目前空壳）
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── admin.py
│   └── migrations/
├── users/                       ← users 子应用（用户中心）
│   ├── __init__.py
│   ├── models.py                ← 用户扩展数据模型
│   ├── views.py                 ← 登录/注册/找回密码逻辑
│   ├── forms.py                 ← 表单验证规则
│   ├── urls.py                  ← users 子路由
│   ├── admin.py                 ← 管理员后台配置
│   ├── static/users/            ← 静态文件（CSS/JS/图片）
│   │   ├── bulma.css            ← Bulma UI 框架样式
│   │   └── js/all.min.js        ← Font Awesome 图标库
│   ├── templates/users/         ← HTML 模板文件夹
│   │   ├── base.html            ← 基础模板（被其他页面继承）
│   │   ├── login.html           ← 登录页
│   │   ├── register.html        ← 注册页
│   │   ├── forget_pwd.html      ← 忘记密码页
│   │   ├── reset_pwd.html       ← 重置密码页
│   │   └── user_profile.html    ← 个人中心页
│   └── migrations/              ← 数据库迁移记录
├── media/images/                ← 用户上传的图片存放处
└── utils/
    └── email_send.py            ← 邮件发送工具模块
```

### 2.1 关键概念：项目 vs 应用

- **项目（Project）**：整个网站，一个 Django 项目可以包含多个「应用」
- **应用（App）**：网站中的一个功能模块。比如你的项目有：
  - `users` 应用：负责用户登录、注册、个人中心
  - `blog` 应用：负责博客文章（目前还是空的）

> 类比：项目 = 一栋商场，应用 = 商场里的不同店铺（服装店、餐厅、电影院）

### 2.2 `manage.py` —— 一切操作的入口

```python
#!/usr/bin/env python
import os
import sys

def main():
    # 设置 Django 使用的配置文件路径
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MyBlog.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed..."
        ) from exc
    execute_from_command_line(sys.argv)
```

**逐行解释：**
- `os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MyBlog.settings')`：告诉 Django「用 `MyBlog/settings.py` 作为配置文件」。相当于给项目指明方向。
- `execute_from_command_line(sys.argv)`：把你输入的命令（如 `runserver`、`makemigrations`）传给 Django 去执行。

**常用命令：**

| 命令 | 作用 |
|------|------|
| `py manage.py runserver` | 启动开发服务器（默认 127.0.0.1:8000） |
| `py manage.py startapp 应用名` | 创建一个新的子应用 |
| `py manage.py makemigrations` | 根据 models.py 生成迁移文件 |
| `py manage.py migrate` | 把迁移文件同步到数据库 |
| `py manage.py createsuperuser` | 创建管理员账号 |

### 2.3 `wsgi.py` —— 部署入口

```python
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MyBlog.settings')
application = get_wsgi_application()
```

**WSGI**（Web Server Gateway Interface）是 Python Web 服务器和 Web 应用之间的标准接口。简单说：**这个文件让你可以把 Django 项目部署到正式的服务器上（如 Nginx + uWSGI）**。

开发阶段用 `runserver` 即可，不用管它。

### 2.4 `__init__.py` —— 让文件夹变成 Python 包

这个文件通常是空的，它的作用是：**告诉 Python「这个文件夹是一个包，可以用 import 导入」**。

如果没有 `__init__.py`，`from users import views` 这样的导入就会报错。

---

## 第三章 settings.py —— 项目配置核心

`settings.py` 是整个项目的「控制面板」，所有配置都在这里。我把它分成几个部分逐个解释：

### 3.1 基础路径

```python
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

`BASE_DIR` 就是项目的**根目录绝对路径**。不管你的代码在哪台电脑上运行，`BASE_DIR` 都能正确指向项目目录。后面的所有路径都基于它来拼接。

例如你电脑上 `BASE_DIR` = `C:\Users\13218\Documents\GitHub\self-test-improvement\博客网站\MyBlog`

### 3.2 安全密钥

```python
SECRET_KEY = ')yy+1c+a0ej1oee_a-dw!&5xd8*@d_c0&l@b_fzz#@o!t-x=5m'
```

这是 Django 用来做加密签名（如 session、CSRF token）的密钥。**正式上线时要换掉且不能泄露。**

### 3.3 调试模式

```python
DEBUG = True
ALLOWED_HOSTS = []
```

- `DEBUG = True`：开发模式，代码有错误会在页面上显示详细信息，方便调试。**上线后必须改成 `False`**
- `ALLOWED_HOSTS = []`：允许哪些域名访问你的网站。开发时空着表示只允许本机访问（127.0.0.1）。上线后要填域名如 `['www.myblog.com']`

### 3.4 自定义认证后端

```python
AUTHENTICATION_BACKENDS = (
    'users.views.MyBackend',
)
```

**这是你自己添加的重要配置！** Django 默认只能用「用户名 + 密码」登录。你写了 `MyBackend` 类来支持**「邮箱 + 密码」也能登录**。这里就是告诉 Django：「用我自定义的认证方式」。

具体原理见第七章 views.py 中 `MyBackend` 的详解。

### 3.5 已安装的应用列表

```python
INSTALLED_APPS = [
    'django.contrib.admin',           # 管理员后台
    'django.contrib.auth',            # 用户认证系统（登录/注册/权限）
    'django.contrib.contenttypes',     # 内容类型框架
    'django.contrib.sessions',         # Session 会话管理
    'django.contrib.messages',         # 消息框架（如"操作成功"提示）
    'django.contrib.staticfiles',      # 管理静态文件（CSS/JS/图片）
    'blog.apps.BlogConfig',            # 你的 blog 应用
    'users.apps.UsersConfig',          # 你的 users 应用
]
```

**关键理解：** 每当你创建一个新应用，必须在这里注册，否则 Django 不知道它的存在。注册方式有两种：
- `'users'`：简写，Django 自动找 `users/apps.py` 中的 `UsersConfig`
- `'users.apps.UsersConfig'`：精确指定配置类，更规范

### 3.6 中间件 Middleware

```python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
```

**中间件是什么？** 它像是请求的「安检通道」。每个请求进来和响应出去时，都会经过中间件的处理。比如：
- `SessionMiddleware`：让你能用 `request.session` 记住用户状态
- `CsrfViewMiddleware`：防止跨站请求伪造攻击（你的模板里看到的 `{% csrf_token %}` 就是它）
- `AuthenticationMiddleware`：让你能用 `request.user` 获取当前登录用户

**一句话理解：** 中间件是一个个钩子，在请求到达视图函数之前/之后执行一些公共逻辑。

### 3.7 模板配置

```python
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.template.context_processors.media',  # 你添加的，处理用户上传文件
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```

- `DIRS`：Django 会在这些目录下搜索模板文件。你这里指定了项目根目录下的 `templates/` 文件夹
- `APP_DIRS: True`：Django 还会在每个应用的 `templates/` 子目录下搜索
- `context_processors`：**模板上下文处理器**，在你渲染模板时自动注入一些变量。比如：
  - `auth`：自动往模板里加 `{{ user }}`（当前用户）和 `{{ perms }}`（权限）
  - `media`：你添加的，让模板能访问用户上传的文件

### 3.8 数据库配置

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}
```

你现在用的是 **SQLite**，它是一个轻量级数据库，数据全存在一个文件（`db.sqlite3`）里。Python 自带支持，无需额外安装。

**开发用 SQLite，上线后一般换成 MySQL 或 PostgreSQL。**

### 3.9 密码验证规则

```python
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},    # 密码不能和用户名太像
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},              # 密码最小长度
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},             # 密码不能太常见
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},            # 密码不能全是数字
]
```

### 3.10 语言和时区

```python
LANGUAGE_CODE = 'zh-hans'    # 中文简体
TIME_ZONE = 'Asia/Shanghai'  # 上海时区（UTC+8）
```

这样 Django 后台界面会显示中文，时间也按中国时间处理。

### 3.11 静态文件和媒体文件

```python
STATIC_URL = '/static/'                    # 静态文件的 URL 前缀
STATICFILES_DIRS = [                        # 开发时静态文件的存放位置
    os.path.join(BASE_DIR, 'static'),
]

MEDIA_URL = '/media/'                       # 用户上传文件的 URL 前缀
MEDIA_ROOT = os.path.join(BASE_DIR, 'media') # 用户上传文件的存放位置
```

**static vs media 的区别：**

| | static（静态文件） | media（媒体文件） |
|---|---|---|
| **是什么** | CSS、JS、logo 等开发时就有的文件 | 用户上传的头像、文章图片等 |
| **示例** | `bulma.css`、`logo.png` | 用户上传的头像图片 |
| **存放位置** | `static/` | `media/` |

### 3.12 邮件配置

```python
EMAIL_HOST = 'smtp.163.com'                        # 你的163邮箱SMTP服务器地址
EMAIL_HOST_USER = "13218798893@163.com"            # 你的邮箱
EMAIL_HOST_PASSWORD = "REQcCfJfRNiXdq8N"           # 邮箱授权码（不是登录密码！）
EMAIL_HOST_PORT = 587                              # SMTP 端口
EMAIL_USE_TLS = True                               # 使用 TLS 加密
EMAIL_USE_SSL = False                              # 不使用 SSL
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
```

> ⚠️ **安全警告**：你的邮箱授权码已经暴露在代码中了！上线前应该从环境变量读取，不能硬编码。

---

## 第四章 models.py —— 数据库模型

模型是**你和数据库交流的翻译官**。你不需要写 SQL 语句，只需写 Python 类，Django 会自动翻译成数据库表。

### 4.1 User 模型（Django 内置）

Django 自带了用户模型 `User`（来自 `django.contrib.auth.models`），包含以下字段：

| 字段 | 含义 |
|------|------|
| `username` | 用户名 |
| `password` | 密码（加密存储） |
| `email` | 邮箱 |
| `first_name` | 名 |
| `last_name` | 姓 |
| `is_active` | 是否激活 |
| `is_staff` | 是否有后台权限 |
| `date_joined` | 注册时间 |

### 4.2 UserProfile —— 扩展用户信息

Django 自带的 User 字段不够用（没有昵称、头像、性别等），于是你创建了 `UserProfile` 来扩展：

```python
class UserProfile(models.Model):
    USER_GENDER_TYPE = (
        ('male', '男'),
        ('female', '女')
    )
    owner = models.OneToOneField(User, on_delete=models.CASCADE, verbose_name='用户')
    nike_name = models.CharField('昵称', max_length=50, blank=True, default='')
    desc = models.TextField('个人简介', max_length=200, blank=True, default='')
    gexing = models.CharField('个性签名', max_length=100, blank=True, default='')
    birthday = models.DateField('生日', null=True, blank=True)
    gender = models.CharField('性别', max_length=6, choices=USER_GENDER_TYPE)
    address = models.CharField('地址', max_length=100, blank=True, default='')
    image = models.ImageField(upload_to='images/%Y/%m', blank=True, null=True, max_length=100, verbose_name='用户头像')

    class Meta:
        verbose_name = '用户数据'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.owner.username
```

#### 逐字段解释：

| 字段 | 类型 | 含义 |
|------|------|------|
| `owner` | `OneToOneField` | **一对一关联** User，每个 User 有且仅有一个 UserProfile |
| `nike_name` | `CharField` | 昵称，最长50字符，非必填 |
| `desc` | `TextField` | 个人简介（长文本），最长200字符 |
| `gexing` | `CharField` | 个性签名，最长100字符 |
| `birthday` | `DateField` | 生日，可以为空 |
| `gender` | `CharField` | 性别，用 choices 限制只能选「男」或「女」 |
| `address` | `CharField` | 地址 |
| `image` | `ImageField` | 用户头像，上传到 `media/images/年/月/` 目录 |

#### 数据库关系详解：

Django 支持三种关系：
- **`OneToOneField`（一对一）**：一个 User 对应一个 UserProfile。就像一个学生对应一张学生证。
- **`ForeignKey`（多对一）**：多个文章对应一个作者。就像一个作者可以写多篇文章。
- **`ManyToManyField`（多对多）**：一个文章可以有多个标签，一个标签可以属于多个文章。

#### 关键参数解释：

- **`on_delete=models.CASCADE`**：当关联的 User 被删除时，UserProfile 也跟着删除（级联删除）
- **`verbose_name='用户'`**：在后台管理页面显示的字段名称（中文）
- **`null=True`**：数据库层面允许存 NULL 值
- **`blank=True`**：表单验证层面允许为空（不填也能提交表单）
- **`choices=USER_GENDER_TYPE`**：限制字段只能选预设值，数据库存 `'male'`/`'female'`，后台显示「男」/「女」
- **`default=''`**：未填时的默认值
- **`upload_to='images/%Y/%m'`**：上传的图片存在 `media/images/年/月/` 下。`%Y` = 四位年份，`%m` = 两位月份

#### Meta 内部类：

```python
class Meta:
    verbose_name = '用户数据'        # 单数形式，后台显示
    verbose_name_plural = verbose_name # 复数形式
```

#### `__str__` 方法：

```python
def __str__(self):
    return self.owner.username
```

决定这个模型对象在后台显示时用什么文字。这里让它显示关联的用户名。

### 4.3 EmailVerifyRecord —— 邮箱验证码

```python
class EmailVerifyRecord(models.Model):
    SEND_TYPE_CHOICES = (
        ('register', '注册'),
        ('forget', '找回密码')
    )
    code = models.CharField('验证码', max_length=20)
    email = models.EmailField('邮箱', max_length=35)
    send_type = models.CharField(choices=SEND_TYPE_CHOICES, max_length=10, default='register')
    send_time = models.DateTimeField('时间', auto_now_add=True)

    class Meta:
        verbose_name = '邮箱验证码'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.code
```

逐字段解释：

| 字段 | 类型 | 含义 |
|------|------|------|
| `code` | `CharField` | 随机生成的8位验证码 |
| `email` | `EmailField` | 收件人邮箱地址 |
| `send_type` | `CharField` | 验证码用途：`'register'`（注册）或 `'forget'`（找回密码） |
| `send_time` | `DateTimeField` | 发送时间，`auto_now_add=True` 表示记录创建时自动填入当前时间 |

**工作原理（以注册为例）：**

1. 用户填写邮箱 → 点击注册
2. 服务器生成一个 8 位随机验证码，存入 `EmailVerifyRecord` 表
3. 发送邮件给用户，邮件里包含链接：`http://127.0.0.1:8000/users/active/验证码/`
4. 用户点击链接 → 服务器根据验证码找到对应记录 → 激活该邮箱对应的账号

---

## 第五章 数据库迁移 Migrations

### 5.1 什么是 Migration？

你写完 `models.py` 后，需要用命令把这些 Python 类变成数据库里真实的表。这个「翻译」过程分两步：

```bash
# 第一步：生成迁移文件（相当于「翻译草稿」）
py manage.py makemigrations users

# 第二步：执行迁移（相当于「正式翻译并写入数据库」）
py manage.py migrate
```

你在 `users/migrations/` 目录下看到的文件（`0001_initial.py`、`0002_...`）就是迁移记录。每个文件记录了一次数据库结构的变更。

### 5.2 迁移文件长什么样？

```python
class Migration(migrations.Migration):
    initial = True                               # 这是第一次迁移
    dependencies = [                              # 依赖其他迁移
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]
    operations = [
        migrations.CreateModel(
            name="UserProfile",
            fields=[ ... ],                       # 要创建的字段
        ),
    ]
```

每次你修改 `models.py`（添加字段、修改字段、删除字段），都需要重新执行 makemigrations 和 migrate。

---

## 第六章 forms.py —— 表单验证

用户在网页上填的表单，提交到服务器后需要验证：密码够不够长？两次密码是否一致？邮箱有没有被注册过？**forms.py 就是做这些验证的。**

### 6.1 LoginForm —— 登录表单

```python
class LoginForm(forms.Form):
    username = forms.CharField(label='用户名', max_length=32, widget=forms.TextInput(attrs={
        'class': 'input', 'placeholder': '用户名/邮箱'
    }))
    password = forms.CharField(label='密码', min_length=6, widget=forms.PasswordInput(attrs={
        'class': 'input', 'placeholder': '密码'
    }))
```

- `forms.CharField`：字符类型字段
- `label='用户名'`：这个字段的标签名
- `max_length=32`：最大32个字符
- `min_length=6`：最少6个字符
- `widget=forms.TextInput(attrs={...})`：指定前端渲染时的 HTML 控件和属性。`class:'input'` 配合 Bulma CSS 框架使用，`placeholder` 是输入框的提示文字
- `forms.PasswordInput`：密码输入框（输入时显示为 `***`）

#### clean_password —— 自定义验证

```python
def clean_password(self):
    username = self.cleaned_data.get('username')
    password = self.cleaned_data.get('password')
    if username == password:
        raise forms.ValidationError('用户名与密码不能相同')
    return password
```

以 `clean_字段名` 命名的函数是 Django 自动调用的**字段验证方法**。`cleaned_data` 是经过 Django 初步验证后的数据（已去除前后空格等）。

### 6.2 RegisterForm —— 注册表单

```python
class RegisterForm(forms.ModelForm):
    email = forms.CharField(label='邮箱', min_length=3, ...)
    password = forms.CharField(label='密码', min_length=6, ...)
    password1 = forms.CharField(label='再次输入密码', min_length=6, ...)

    class Meta:
        model = User
        fields = ('email', 'password')
```

这里用的是 `forms.ModelForm`（而不是 `forms.Form`）。区别：

| | `forms.Form` | `forms.ModelForm` |
|---|---|---|
| 和数据库关系 | 无关，纯粹的验证 | 直接关联某个 Model |
| 保存数据 | 需要手动处理 | 调用 `form.save()` 直接保存到数据库 |

你的注册表单关联了 `User` 模型，只用到 `email` 和 `password` 两个字段。

#### clean_email —— 检查邮箱是否已注册

```python
def clean_email(self):
    email = self.cleaned_data.get('email')
    exists = User.objects.filter(email=email).exists()
    if exists:
        raise forms.ValidationError('用户名已经存在')
    return email
```

`User.objects.filter(email=email).exists()` 去数据库查这个邮箱是否已被注册。

### 6.3 ForgetpwdForm —— 忘记密码表单

```python
class ForgetpwdForm(forms.Form):
    email = forms.EmailField(label='请输入注册邮箱地址', min_length=4, widget=forms.EmailInput(attrs={
        'class': 'input', 'placeholder': '用户名/邮箱'
    }))
```

用户只需输入邮箱，系统发一封修改密码的邮件。

### 6.4 ModifyPwdForm —— 修改密码表单

```python
class ModifyPwdForm(forms.Form):
    password = forms.CharField(label='输入新密码', min_length=6, widget=forms.PasswordInput(...))
```

用户从邮件链接点过来之后，输入新密码。

---

## 第七章 views.py —— 视图函数（核心业务逻辑）

视图函数是整个 Django 的**核心**——它接收用户的请求，处理业务逻辑，然后返回响应（HTML页面或文字）。

### 7.1 导入语句解释

```python
from django.shortcuts import render, HttpResponse, redirect
# render:      渲染 HTML 模板并返回给用户
# HttpResponse: 返回纯文本（如 '注册成功'）
# redirect:     重定向到另一个 URL

from django.contrib.auth import authenticate, login
# authenticate: 验证用户名和密码是否正确
# login:        把用户标记为「已登录」状态（写入 session）

from django.contrib.auth.backends import ModelBackend
# Django 默认的认证后端，你继承了它来扩展功能

from django.contrib.auth.models import User
# Django 内置的 User 模型

from django.db.models import Q
# Q 对象，用于构建复杂的查询条件（如「用户名=xxx 或 邮箱=xxx」）

from .models import EmailVerifyRecord, UserProfile
# 从当前应用的 models 导入你定义的模型

from utils.email_send import send_register_email
# 导入你写的邮件发送函数

from .forms import LoginForm, RegisterForm, ForgetpwdForm, ModifyPwdForm
# 导入你定义的表单类

from django.contrib.auth.hashers import make_password
# 把明文密码变成加密后的密文（不能存明文！）

from django.contrib.auth.decorators import login_required
# 装饰器：要求用户必须登录才能访问某个页面
```

### 7.2 MyBackend —— 自定义认证后端（支持邮箱登录）

```python
class MyBackend(ModelBackend):
    '''邮箱登录注册'''
    def authenticate(self, request, username=None, password=None, **kwargs):
        try:
            # Q(username=username) | Q(email=username)
            # 意思：用户名=输入值  或  邮箱=输入值
            user = User.objects.get(Q(username=username) | Q(email=username))
            if user.check_password(password):
                return user
        except Exception as e:
            return None
```

**原理：**
- Django 默认只能用 `username` 登录
- 你重写的 `authenticate` 方法会同时匹配 `username` 和 `email` 字段
- `Q(username=username) | Q(email=username)`：`|` 表示「或」，在数据库里查「用户名匹配 OR 邮箱匹配」
- `user.check_password(password)`：验证密码（自动解密对比，密码不是明文存的）
- 返回 `user` 对象表示验证通过，返回 `None` 表示失败

### 7.3 login_view —— 登录视图

```python
def login_view(request):
    if request.method != 'POST':
        form = LoginForm()     # GET 请求：显示空白登录表单
    else:
        form = LoginForm(request.POST)  # POST 请求：用提交的数据创建表单
        if form.is_valid():             # 调用表单验证
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)    # 把用户标记为已登录
                return redirect('users:user_profile')  # 跳转到个人中心
            else:
                return HttpResponse('账号或密码错误！')
    return render(request, 'users/login.html', {'form': form})
```

**逐行流程图解：**

```
用户访问 /users/login
        ↓
   是 POST 请求吗？
   ↙No          ↘Yes
显示空白表单    用提交数据创建表单对象
                     ↓
               form.is_valid()？
               ↙No         ↘Yes
         返回表单+错误提示  取出用户名和密码
                                ↓
                         authenticate() 验证
                         ↙失败        ↘成功
                  返回"账号密码错误"  login()标记登录
                                     重定向到个人中心
```

关键概念：
- `request.method`：区分 GET（查看页面）和 POST（提交表单）
- `form.is_valid()`：触发你在 forms.py 中写的所有验证规则
- `form.cleaned_data`：验证通过后的干净数据（字典格式）
- `login(request, user)`：Django 会把用户信息存入 session，后续请求都能通过 `request.user` 获取当前用户
- `redirect('users:user_profile')`：`'users:user_profile'` 是命名 URL（在 urls.py 中定义的 `name='user_profile'`）

### 7.4 register —— 注册视图

```python
def register(request):
    if request.method != 'POST':
        form = RegisterForm()
    else:
        form = RegisterForm(request.POST)
        if form.is_valid():
            new_user = form.save(commit=False)    # ① 生成 User 对象但不保存
            new_user.set_password(form.cleaned_data.get('password'))  # ② 加密密码
            new_user.username = form.cleaned_data.get('email')        # ③ 用邮箱当用户名
            new_user.save()                        # ④ 保存到数据库
            send_register_email(form.cleaned_data.get('email'), 'register')  # ⑤ 发激活邮件
            return HttpResponse('注册成功')
    return render(request, 'users/register.html', {'form': form})
```

**关键步骤详解：**

- **`form.save(commit=False)`**：`RegisterForm` 是 `ModelForm`，可以直接 `.save()` 保存到数据库。但 `commit=False` 意思是「先创建对象，别急着存」，因为还要做一些额外处理。
- **`set_password()`**：绝对不能把密码明文存数据库！`set_password()` 会用哈希算法加密密码。
- **`new_user.username = email`**：因为你用邮箱注册，所以把邮箱也设为 username。
- **`send_register_email()`**：调用你写的邮件发送函数（详见第十一章）。

### 7.5 active_user —— 激活用户

```python
def active_user(request, active_code):
    all_records = EmailVerifyRecord.objects.filter(code=active_code)
    if all_records:
        for record in all_records:
            email = record.email
            user = User.objects.get(email=email)
            user.is_active = True   # 激活用户
            user.save()
    else:
        return HttpResponse('链接有误')
    return redirect('users:login')
```

**`active_code` 来自哪里？**
它来自 URL：`path('active/<active_code>/', views.active_user, name='active_user')`。`<active_code>` 是一个 URL 参数，Django 自动捕获并传给视图函数。

用户点击邮件中的 `http://127.0.0.1:8000/users/active/xxxxx/` 时，`xxxxx` 就会被赋给 `active_code`。

### 7.6 forget_pwd —— 忘记密码

```python
def forget_pwd(request):
    if request.method == 'GET':
        form = ForgetpwdForm()
    elif request.method == 'POST':
        form = ForgetpwdForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            exists = User.objects.filter(email=email).exists()
            if exists:
                send_register_email(email, 'forget')  # 发送重置密码邮件
                return HttpResponse('邮箱已经发送请查收')
            else:
                return HttpResponse('邮箱还未注册，请前往注册')
    return render(request, 'users/forget_pwd.html', {'form': form})
```

和注册逻辑类似：用户输入邮箱 → 检查邮箱是否存在 → 发送重置密码邮件。

### 7.7 forget_pwd_url —— 重置密码确认

```python
def forget_pwd_url(request, active_code):
    if request.method != 'POST':
        form = ModifyPwdForm()
    else:
        form = ModifyPwdForm(request.POST)
        if form.is_valid():
            record = EmailVerifyRecord.objects.get(code=active_code)
            email = record.email
            user = User.objects.get(email=email)
            user.username = email
            user.password = make_password(form.cleaned_data.get('password'))
            user.save()
            return HttpResponse('修改成功')
    return render(request, 'users/reset_pwd.html', {'form': form})
```

用户点击邮件链接 → 填写新密码 → `make_password()` 加密新密码 → 保存。

### 7.8 user_profile —— 个人中心

```python
@login_required(login_url='users:login')
def user_profile(request):
    user = User.objects.get(username=request.user)
    return render(request, 'users/user_profile.html', {'user': user})
```

- **`@login_required(login_url='users:login')`**：装饰器，检查用户是否登录。如果没登录，自动跳转到登录页。
- **`request.user`**：获取当前登录的用户对象。
- **`render(request, '模板路径', {'user': user})`**：渲染模板，第三个参数是把数据传给模板。在模板中就可以用 `{{ user }}` 来访问。

---

## 第八章 urls.py —— URL 路由

### 8.1 项目级路由 `MyBlog/urls.py`

```python
from django.contrib import admin
from django.urls import include, path
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),              # 管理员后台
    path('users/', include('users.urls')),         # users 应用的所有 URL
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

- **`path('users/', include('users.urls'))`**：所有以 `users/` 开头的 URL 都交给 `users/urls.py` 处理
- **`+ static(...)`**：开发阶段，让 Django 能直接提供静态文件和用户上传文件的访问

### 8.2 子路由 `users/urls.py`

```python
from django.urls import path
from . import views

app_name = 'users'   # 命名空间，用于模板中的 {% url 'users:xxx' %}

urlpatterns = [
    path('login', views.login_view, name='login'),
    path('register', views.register, name='register'),
    path('active/<active_code>/', views.active_user, name='active_user'),
    path('forget_pwd/', views.forget_pwd, name='forget_pwd'),
    path('forget_pwd_url/<active_code>/', views.forget_pwd_url, name='forget_pwd_url'),
    path('user_profile/', views.user_profile, name='user_profile'),
]
```

#### URL 映射表：

| URL 模式 | 视图函数 | 完整访问地址 | 用途 |
|----------|---------|-------------|------|
| `login` | `login_view` | `/users/login` | 登录页 |
| `register` | `register` | `/users/register` | 注册页 |
| `active/<active_code>/` | `active_user` | `/users/active/xxxx/` | 激活邮箱 |
| `forget_pwd/` | `forget_pwd` | `/users/forget_pwd/` | 忘记密码 |
| `forget_pwd_url/<active_code>/` | `forget_pwd_url` | `/users/forget_pwd_url/xxxx/` | 重置密码 |
| `user_profile/` | `user_profile` | `/users/user_profile/` | 个人中心 |

#### `<active_code>` 是什么？

尖括号括起来的叫 **URL 路径参数**。Django 会自动把 URL 中这部分捕获并传递给视图函数。

例如：用户访问 `/users/active/abc123/`，视图函数 `active_user(request, active_code='abc123')` 就会被调用。

#### `name='xxx'` 的作用：

给 URL 起个名字，在模板中可以用 `{% url 'users:login' %}` 来引用，而不是写死 `/users/login`。好处是万一 URL 变了，只需改 urls.py 一处。

---

## 第九章 templates —— HTML 模板

### 9.1 模板语法入门

Django 模板使用以下语法：

| 语法 | 用途 | 示例 |
|------|------|------|
| `{{ 变量 }}` | 输出变量值 | `{{ user.username }}` |
| `{% 标签 %}` | 执行逻辑（if/for/url等） | `{% if user %}...{% endif %}` |
| `{% block 名称 %}` | 定义可被子模板替换的区块 | `{% block content %}...{% endblock %}` |

### 9.2 base.html —— 模板继承

```html
<!DOCTYPE html>
{% load static %}                        <!-- 加载 static 标签库 -->
<html lang="en">
<head>
    <meta charset="utf-8">
    {% block head %}<title>登录页面</title>{% endblock head %}
    <link rel="stylesheet" href="{% static 'users/bulma.css' %}">
    <!-- {% static '路径' %} → 生成 /static/users/bulma.css -->
</head>
<body>
<div class="container">
    {% block content %}
    {% endblock content %}
</div>
</body>
</html>
```

**模板继承（Template Inheritance）**的核心思想：

`base.html` 定义了页面的整体框架（HTML 结构、CSS 引用），然后用 `{% block %}` 挖了几个「洞」。子模板继承 `base.html` 后，只需要填这些「洞」就行。

### 9.3 login.html —— 子模板

```html
{% extends 'users/base.html' %}          <!-- 继承 base.html -->

{% block head %}
    <title>登录页面</title>                <!-- 覆盖 base.html 的 head 块 -->
{% endblock head %}

{% block content %}                       <!-- 覆盖 base.html 的 content 块 -->
    <div class="box">
        <h1 class="...">轻编程<br><span class="tag">登录</span></h1>
        <hr>
        <form action="" method="post">     <!-- action="" 表示提交到当前 URL -->
            {% csrf_token %}               <!-- CSRF 安全令牌，必须写！ -->
            {{ form.as_p }}                <!-- 渲染表单，每个字段包在 <p> 标签里 -->
            <a href="{% url 'users:forget_pwd' %}">忘记密码？</a>
            <input type="submit" value="登 录">
        </form>
    </div>
{% endblock content %}
```

#### 关键模板语法详解：

**`{% csrf_token %}`**
- CSRF = 跨站请求伪造（Cross-Site Request Forgery）
- Django 的安全机制：每个 POST 表单都必须有这个 token，否则请求被拒绝
- 它在 HTML 中生成一个隐藏的 `<input type="hidden">`，包含随机 token

**`{{ form.as_p }}`**
- 把 Django Form 对象渲染成 HTML
- `.as_p`：每个字段用 `<p>` 标签包裹
- `.as_table`：用表格渲染
- `.as_ul`：用列表渲染

**`{% url 'users:forget_pwd' %}`**
- 根据 urls.py 中定义的 `name` 生成对应的 URL
- `'users:forget_pwd'` = `app_name:name` = `/users/forget_pwd/`

### 9.4 user_profile.html —— 个人中心

```html
{% if user.userprofile.nike_name %}
    <span>{{ user.userprofile.nike_name }}</span>
{% else %}
    <span>{{ user.username }}</span>
{% endif %}
```

模板中可以通过关联关系访问数据：
- `user.userprofile`：通过 OneToOneField 的反向关联，获取 User 对应的 UserProfile
- `{% if profile and profile.image %}`：判断用户是否上传了头像

---

## 第十章 admin.py —— 后台管理

Django 自带一个强大的管理后台，访问 `/admin/` 即可进入。

### 10.1 注册模型到后台

```python
from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from .models import UserProfile, EmailVerifyRecord

# 先把内置的 User 取消注册
admin.site.unregister(User)

# 定义 UserProfile 在 User 编辑页面中的显示方式
class UserProfileInline(admin.StackedInline):
    model = UserProfile   # 关联的模型

# 把 UserProfile 嵌入到 User 的编辑页面
class UserProfileAdmin(UserAdmin):
    inlines = [UserProfileInline]

# 重新注册 User（带上了 UserProfile）
admin.site.register(User, UserProfileAdmin)

# 注册 EmailVerifyRecord
@admin.register(EmailVerifyRecord)
class Admin(admin.ModelAdmin):
    list_display = ('code',)   # 列表页显示 code 字段
```

#### 关键理解：

- **`admin.site.register()`**：把模型注册到后台，这样才能在 `/admin/` 页面看到和管理它
- **`StackedInline`**：让 UserProfile 以「纵向堆叠」方式嵌入 User 编辑页。`TabularInline` 则是表格形式。
- **`@admin.register(EmailVerifyRecord)`**：装饰器写法，等价于 `admin.site.register(EmailVerifyRecord, Admin)`

### 10.2 如何使用后台？

```bash
# 先创建超级管理员账号
py manage.py createsuperuser

# 然后访问 http://127.0.0.1:8000/admin/ 登录
```

---

## 第十一章 邮件发送模块

### 11.1 email_send.py 完整解析

```python
from users.models import EmailVerifyRecord
from django.core.mail import send_mail
import random
import string

def random_str(randomlength=8):
    '''生成 8 位随机字符串（字母+数字）'''
    chars = string.ascii_letters + string.digits   # 所有大小写字母 + 0-9
    strcode = ''.join(random.sample(chars, randomlength))
    return strcode

def send_register_email(email, send_type='register'):
    # ① 创建 EmailVerifyRecord 记录，存到数据库
    email_record = EmailVerifyRecord()
    code = random_str()              # 生成随机验证码
    email_record.code = code
    email_record.email = email
    email_record.send_type = send_type
    email_record.save()              # 保存到数据库

    # ② 根据类型发送不同内容的邮件
    if send_type == 'register':
        email_title = '博客的注册激活链接'
        email_body = f'请点击以下链接激活账号：http://127.0.0.1:8000/users/active/{code}/'
        send_status = send_mail(
            email_title,                          # 邮件标题
            email_body,                           # 邮件正文
            '13218798893@163.com',                # 发件人（你的邮箱）
            [email],                              # 收件人列表
        )

    elif send_type == 'forget':
        email_title = '找回密码链接'
        email_body = f'请点击以下链接修改密码：http://127.0.0.1:8000/users/forget_pwd_url/{code}/'
        send_status = send_mail(email_title, email_body, '13218798893@163.com', [email])
```

#### 完整流程图：

```
用户注册 → 输入邮箱
        ↓
register() 视图函数
        ↓
send_register_email(email, 'register')
        ↓
生成 8 位随机码（如 "aB3xK9mQ"）
        ↓
存入 EmailVerifyRecord 表
| code      | email           | send_type | send_time          |
|-----------|-----------------|-----------|--------------------|
| aB3xK9mQ  | user@qq.com     | register  | 2026-07-09 10:00   |
        ↓
send_mail() 发邮件
邮件内容："点击 http://127.0.0.1:8000/users/active/aB3xK9mQ/ 激活"
        ↓
用户点击链接 → active_user视图 → 验证code → 激活账号
```

---

## 第十二章 完整的请求流程总结

让我用「用户注册」这个完整的例子，把上面的所有知识串起来：

```
1. 用户在浏览器输入 http://127.0.0.1:8000/users/register
        ↓
2. Django 收到 GET 请求
        ↓
3. MyBlog/urls.py: 匹配 'users/' → 交给 users/urls.py
        ↓
4. users/urls.py: 匹配 'register' → 调用 views.register()
        ↓
5. register() 判断 request.method != 'POST' → 创建空白 RegisterForm
        ↓
6. render(request, 'users/register.html', {'form': form})
        ↓
7. Django 找到 register.html 模板
        ↓
8. {% extends 'users/base.html' %} → 继承 base.html 框架
        ↓
9. {{ form.as_p }} → RegisterForm 渲染成 HTML：邮箱输入框 + 密码框 + 确认密码框
        ↓
10. {% csrf_token %} → 生成隐藏的 CSRF token
        ↓
11. 完整的 HTML 页面返回给浏览器，用户看到注册表单
        ↓
12. 用户填写邮箱、密码 → 点击「注册」
        ↓
13. 浏览器发送 POST 请求（表单数据在请求体中）
        ↓
14. 再次进入 register()，这次 request.method == 'POST'
        ↓
15. RegisterForm(request.POST) → 用提交的数据创建表单对象
        ↓
16. form.is_valid() → 触发验证：
    - clean_email()：检查邮箱是否已注册
    - clean_password1()：检查两次密码是否一致
    - Django 内置验证：密码最小长度等
        ↓
17. 验证通过 → form.save(commit=False) → 创建 User 对象（还没存）
        ↓
18. set_password() → 加密密码
        ↓
19. new_user.save() → 存入数据库 User 表
        ↓
20. send_register_email() → 生成验证码 → 存 EmailVerifyRecord 表 → 发邮件
        ↓
21. return HttpResponse('注册成功')
        ↓
22. 用户收到邮件 → 点击激活链接 → active_user() → is_active=True → 激活完成
```

---

## 附录：常用命令速查表

| 命令 | 作用 |
|------|------|
| `py manage.py runserver` | 启动开发服务器 |
| `py manage.py startapp 名称` | 创建新应用 |
| `py manage.py makemigrations` | 生成迁移文件 |
| `py manage.py migrate` | 同步数据库 |
| `py manage.py createsuperuser` | 创建管理员账号 |
| `py manage.py shell` | 进入 Django 交互式 shell |
| `py manage.py dbshell` | 进入数据库命令行 |

---

> 本教程完。建议对照你自己的项目代码反复阅读，每一行代码都要理解它在「注册→登录→个人中心」整个流程中扮演了什么角色。理解了这个项目，你就掌握了 Django 的核心用法。
