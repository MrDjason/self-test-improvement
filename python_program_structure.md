# Python 程序结构

## 一、单个Python程序结构
### 顺序 
`1.` **Py文件说明**  
`2.` **导入模块**  
`3.` **全局变量**  
`4.` **函数**  
`5.` **类**  
`6.` **主程序入口**  
`7.` **触发主程序**

### 结构
```python
# 1. 文档字符串（可选，但推荐）
"""
模块功能描述：比如"处理用户数据的工具类"
作者、版本、日期等元信息
"""

# 2. 导入模块（标准库 → 第三方库 → 自定义模块）
import os  # 标准库
import pandas as pd  # 第三方库
from my_utils import format_data  # 自定义模块

# 3. 全局变量/常量（全大写命名，如MAX_SIZE）
MAX_RETRY = 3
DEFAULT_PATH = "./data"

# 4. 工具函数/辅助函数（被其他函数/类调用的通用功能）
def check_file_exists(path):
    """检查文件是否存在"""
    return os.path.exists(path)

# 5. 类定义（如果需要面向对象编程）
class UserProcessor:
    """处理用户数据的类"""
    def __init__(self, username):
        self.username = username
    
    def get_user_info(self):
        # 具体实现
        pass

# 6. 主程序入口（核心逻辑，通过if __name__ == "__main__"触发）
def main():
    # 调用上面定义的函数/类，实现业务逻辑
    if check_file_exists(DEFAULT_PATH):
        processor = UserProcessor("test_user")
        processor.get_user_info()

# 7. 触发主程序（确保脚本被直接运行时才执行main，被导入时不执行）
if __name__ == "__main__":
    main()
```

## 二、项目级Python程序的结构

### (1) 结构
```Python
my_project/
├── README.md
├── requirements.txt
├── .gitignore
├── setup.py / pyproject.toml（可选）
├── config/
│   └── config.yaml
├── data/
│   └── raw_data.csv
├── scripts/
│   └── run_experiment.py
├── tests/
│   └── test_main.py
├── my_project/  # 主模块
│   ├── __init__.py
│   ├── main.py
│   ├── utils.py
│   ├── modules/
│   │   ├── __init__.py
│   │   └── xxx_module.py
│   └── services/
│       └── xxx_service.py
```
### (2) 各部分说明
| 文件/文件夹             | 作用说明                                   |
| -----------------------| ------------------------------------------|
| `README.md`            | 项目说明文档（功能、安装、使用方法等）|
| `requirements.txt`     | 第三方依赖包清单（如 `pandas==1.5.3`）|
| `scripts/`             | 辅助脚本（实验运行、数据初始化等工具）|
| `tests/`               | 测试用例（推荐用 `pytest` 框架）|
| `my_project/`          | 主程序包（核心业务逻辑的根目录）|
| `my_project/modules/`  | 功能模块分层（如文本处理、数据库操作等子功能） |
| `my_project/services/` | 业务逻辑的服务封装（对接外部依赖、复杂流程） |
| `config/`              | 配置文件目录（如 `config.yaml` 集中存参数）|
| `data/`                | 数据目录（样例数据、输入/输出结果等）|

### (3) 按功能划分主程序包
#### 模块划分：拆分成 `modules/`文件下的子模块
- **公共工具函数**：`utils.py`
- **配置管理**：`config/config.yaml`
- **数据加载**：`data_loader.py`
#### 使用配置文件管理参数
- **配置参数不要硬编码。最常见的做法是使用`YAML`**：
```Python
# config/config.yaml
input_file: ./data/input.csv
output_dir: ./data/results/
batch_size: 32
```
- **然后用`PyYAML`加载**
```Python
import yaml

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)
print(config['batch_size'])  # 32
```
- **优势是：便于复现、易于共享、方便修改**
#### 使用`logging`代替`print`管理日志
- **使用标准 logging 模块**：
```Python
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("开始处理数据")
```
- **可以写入日志文件**
- **可控输出等级（DEBUG/INFO/ERROR）**
- **更方便线上调试**
#### 模块化结构 + `__main__.py`支持命令行运行
```Python
def run():
    # 主逻辑
    ...

if __name__ == "__main__":
    run()
```
- **或者更专业地支持 CLI**：
```Python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/config.yaml')
args = parser.parse_args()
```
- **这样就可以通过命令行运行项目**：
```cmd
python main.py --config=config/my_config.yaml
```
#### 引入单元测试（pytest）
- **推荐结构**：

```Python
tests/
├── __init__.py
├── test_utils.py
├── test_service.py
```
- **使用 pytest**：
```Python
# test_utils.py
from my_project.utils import add

def test_add():
    assert add(1, 2) == 3
```
- **运行**：
```Python
pytest tests/
```
## 进阶建议（可选）：
- **想打造更专业的 Python 项目，还可以加上***：  

|工具 / 技术|作用|
|:---|:---|
|black + isort + flake8|格式化和风格检查|
|pre-commit	Git|提交前自动检查|
|setup.py / pyproject.toml|支持 pip 安装、自定义命令|
|Poetry|虚拟环境 + 依赖管理 + 打包一体化|
|makefile / task.py|一键运行任务（如训练/部署）|
|Dockerfile|容器化部署，标准交付|