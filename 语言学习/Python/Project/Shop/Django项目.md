# 项目需求分析

> 需求分析原因
> - 项目中，**需求驱动开发**，即开发人员需要以需求为目标来实现业务逻辑。
> 
> 需求分析方式
> - 企业中，借助产品原型图分析需求。
> - 需求分析完后，前端按照产品原型图开发前端页面，后端开发对应的业务及响应处理。
> - 提示：我们现在假借示例网站作为原型图来分析需求。
> 
> 需求分析内容
> - 页面及页面上的业务逻辑。
> - 归纳业务逻辑并划分模块。

![人员调动](./res/人员.png '人员调动')

### 1. 项目主要页面介绍

#### 1.1.2 项目架构设计

**1.项目开发模式**  

|选项|技术选型|
|:---:|:---:|
|开发模式|前后端分离|
|后端框架|Django|
|前端框架|Vue.js|

> 说明：
> - 页面需要局部刷新，通过使用Vue.js来实现

**2.项目架构设计**

### 1.2 工程创建和配置

#### 1.2.1 创建工程

#### 1.2.2配置开发目录

```
meiduo_mall/                          
├── apps/                              # 子应用包
│   └── __init__.py                    # 标识为Python包，使其可被导入
├── libs/                              # 第三方库包：存放手动引入/修改的第三方工具库
│   └── __init__.py                    
├── meiduo_mall/                       # 项目核心配置
│   ├── __init__.py                    
│   ├── settings.py                    # 全局配置
│   ├── urls.py                        # 根路由
│   └── wsgi.py                        # WSGI接口
├── templates/                         # 模板文件夹
├── utils/                             # 公共工具包
│   └── __init__.py                    
└── manage.py                          # Django管理脚本
```

| 目录/文件       | 核心作用                                                                 |
|-----------------|--------------------------------------------------------------------------|
| `apps/`         | 按业务拆分子应用（如用户、商品、订单），让代码更清晰、易维护               |
| `libs/`         | 存放项目依赖的第三方工具库（区别于虚拟环境`venv`中的自动安装依赖）         |
| `meiduo_mall/`  | 项目的“大脑”，管理全局配置、路由规则和部署接口                           |
| `templates/`     | 存放Django模板（HTML），用于渲染动态页面                                 |
| `utils/`         | 存放通用工具函数/类（如IP获取、订单号生成），避免在各子应用中重复造轮子     |
| `manage.py`     | Django命令行入口，通过`python manage.py <命令>`执行项目管理操作            |

#### 1.2.3 运行前端

进入到前端文件夹

```python
# cd 前端文件夹
python3 -m http.server 8080
```

#### 1.2.4 配置Mysql数据库

**1.新建Mysql数据库**

> 1.新建Mysql数据库：meiduo_mall

```mysql
ccreate database meiduo_mall charset=utf8;
```

> 2.新建Mysql用户  
> identified by '123456'设置密码  

```mysql
create user dr3 identified by '123456';
```

> 3.授权itcast用户访问`meiduo_mall`数据库  
> grant all 授予所有权限  
> meiduo_mall.* 此数据库下所有表  
> 'dr3'@'%' 权限授予dr3，%表示允许用户从任何主机访问  

```mysql
grant all on meiduo_mall.* to 'dr3'@'%';
```

> 4.授权结束后刷新特权  
> 让 MySQL 立即加载最新的权限配置。如果不执行，可能需要重启 MySQL 服务才能生效  

```mysql
flush privileges;
```

**2.配置数据库**  

> settings.py

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql', # 数据库引擎
        'HOST': '127.0.0.1', # 数据库主机
        'PORT': 3306, # 数据库端口
        'USER': 'dr3', # 数据库用户名
        'PASSWORD': '123456', # 数据库用户密码
        'NAME': 'meiduo_mall', # 数据库名字
    }
}
```

#### 1.2.5 配置Redis数据库

**1.安装django-redis扩展包**

[django-redis使用说明文档](http://django-redis-chs.readthedocs.io/zh_CN/latest/)

> 在settings.py

```python 
pip install django-redis
```

**2.配置Redis数据库**

```python
 # 缓存配置
CACHES = {
    "default": { # 预留
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/0",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    },
        "session": { # 用于保存session数据
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/1",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    },
}

# Session存储配置
SESSION_ENGINE = "django.contrib.sessions.backends.cache"
SESSION_CACHE_ALIAS = "default"
```

LOCATION后面跟着的0、1是Redis数据库的编号（Redis默认内置16个数据库，编号0-15）。

> default:  
> - 默认的Redis配置项，采用0号Redis库  
> 
> session:  
> - 状态保持的Redis配置项，采用1号Redis库  
> 
> SESSION_ENGINE:  
> - 修改`session存储机制`使用Redis保存  
> 
> SESSION_CACHE_ALIAS:  
> - 使用名为session的Redis配置项存储session数据  
> 配置完成后运行程序测试结果  

#### 1.2.6 配置工程日志

[日志文档](http://docs.djangoproject.com/en/1.11/topics/logging/)

**1.配置工程日志**

> settings.py

```python
import os
# 从Django配置中导入项目根目录常量（需确保项目已定义BASE_DIR）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 日志配置
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,  # 是否禁用已存在的日志器
    # 日志格式配置
    'formatters': {
        # 详细格式
        'verbose': {
            'format': '%(levelname)s %(asctime)s %(module)s %(lineno)d %(message)s'
        },
        # 简单格式
        'simple': {
            'format': '%(levelname)s %(module)s %(lineno)d %(message)s'
        },
    },
    # 日志过滤规则
    'filters': {
        # 仅在Django的DEBUG模式为True时输出日志
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        },
    },
    # 日志处理方式（终端/文件）
    'handlers': {
        # 终端输出日志
        'console': {
            'level': 'INFO',
            'filters': ['require_debug_true'],
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        # 文件输出日志（按大小轮转）
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs/meiduo.log'),  # 日志文件路径
            'maxBytes': 300 * 1024 * 1024,  # 单个日志文件最大300MB
            'backupCount': 10,  # 最多保留10个日志备份文件
            'formatter': 'verbose'
        },
    },
    # 日志器核心配置
    'loggers': {
        # 针对django框架的日志器
        'django': {
            'handlers': ['console', 'file'],  # 同时输出到终端和文件
            'propagate': True,  # 是否向上传递日志信息
            'level': 'INFO',  # 日志接收的最低级别
        },
    },
}
```

**2.准备日志文件目录**

```python
meiduo_mall/logs/meiduo.log
```

**3.日志记录器的使用**

不同的应用程序所定义的日志等级可能会有所差别，分的详细点会包含以下几个等级：

- FATAL/CRITICAL = 重大的，危险的
- **ERROR = 错误**
- **WARNING = 警告**
- **INFO = 信息**
- **DEBUG = 调试**
- NOTSET = 没有设置  

设置的日志在Django里接受的Logger的最低level为INFO，则DEBUG的信息不会显示

```python
import logging

# 创建日志记录器
logger = logging.getLogger('django')

# 输出日志
logger.debug('测试logging模块debug')
logger.info('测试logging模块info')
logger.error('测试logging模块error')
```

**4.Git记录工程日志**

#### 1.2.7 配置访问域名

设置访问域名  

**虚拟机**

|位置|域名|
|:---:|:---:|
|前端|www.meiduo.site|

编辑`/etc/hosts`文件，可以设置本地域名  

```bash
sudo vim /etc/hosts
```

在文件中增加信息  

```bash
127.0.0.1 www.meiduo.site
``` 

**Windows**

找到hosts  

```bash
C:\Windows\System32\drivers\etc\hosts
```

添加域名映射，在文件末尾增添

```bash
# 示例：将 www.meiduo.site 指向本地（127.0.0.1）或你的服务器IP
127.0.0.1 www.meiduo.site
# 如果是指向虚拟机/远程服务器，替换为对应IP，比如：
# 192.168.1.100 www.meiduo.site
```

设置ALLOWED_HOSTS  

```python
# 允许哪些主机访问
ALLOWED_HOSTS = ['www.meiduo.site', '127.0.0.1']
```

## 2. 用户注册

### 2.1 用户模型类

**创建用户模块应用**

> 在`apps`包下创建应用`users`

```bash
# apps文件夹内
python ../manage.py startapp users
```

**注册用户模块应用**



### 2.2 用户注册业务实现

注册子应用

```python
INSTALLED_APPS = [


]
```