## Nuitka 基础与工作原理

Nuitka 是一个 Python 到 C/C++ 的编译器。与 PyInstaller、cx_Freeze 等仅仅将 Python 解释器和脚本源码压缩打包成包的“打包器”不同，Nuitka 会先将 Python 源码翻译成 C 代码，然后调用系统的 C/C++ 编译器（如 GCC/Clang 或 MSVC）将其编译为原生机器码机器二进制。

* **性能提升**：计算密集型代码运行速度通常会有明显提升。
* **高强度源码保护**：代码直接被编译成二进制机器码，难以被反编译还原。
* **依赖自动分析**：内置成熟插件体系（如 PySide6、PyQt、NumPy、Torch 等），精准捕获 C/C++ DLL 依赖树。

---

## 一、安装与环境准备

### 1. 安装 Nuitka

在当前 Python 环境（建议在虚拟环境中）执行：

```powershell
pip install nuitka zstandard

```

> `zstandard` 用于提供高压缩比支持（单文件打包必须依赖该库）。

### 2. 编译器环境

Windows 下编译需要 C 编译器：

* **自动下载（推荐）**：首次编译时，Nuitka 检测到没有编译器会自动询问 `Is it OK to download and put it in local user cache?`，直接回车输入 `Yes`，它会自动下载预构建的 MinGW64（GCC）。
* **已安装 Visual Studio**：如果已安装带 C++ 开发环境的 Visual Studio（MSVC），Nuitka 会自动识别并优先调用。

---

## 二、常用核心命令

### 1. 文件夹模式打包（`--standalone`，推荐开发测试使用）

将程序及所有依赖解压收集到一个文件夹内，排查问题最直观，且启动速度最快。

```powershell
nuitka --standalone main.py

```

* **输出位置**：在当前目录下生成 `main.dist` 文件夹，双击其中的 `main.exe` 即可运行。

### 2. 单文件模式打包（`--onefile`）

将整个程序和依赖打包压缩成一个单独的 `.exe` 文件：

```powershell
nuitka --onefile main.py

```

* **输出位置**：在当前目录下生成 `main.exe`（以及一个用于临时构建的 `main.build` 目录）。

### 3. 隐藏控制台黑窗口（GUI 程序专用）

关闭黑色的命令提示符控制台窗口：

```powershell
nuitka --standalone --windows-console-mode=disable main.py

```

*(如果是较旧版本参数，也可以直接写 `--windows-disable-console`)*

---

## 三、常用参数与插件说明

| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `--standalone` | 独立文件夹模式，包含依赖与运行时环境 | `--standalone` |
| `--onefile` | 单个可执行文件模式 | `--onefile` |
| `--enable-plugin=<插件名>` | 激活针对特定复杂库的官方插件（如 Qt 系列） | `--enable-plugin=pyside6` |
| `--windows-console-mode=disable` | 运行时不显示黑色的终端控制台窗口 | `--windows-console-mode=disable` |
| `--windows-icon-from-ico=<图标路径>` | 为生成的 exe 文件设置应用图标 | `--windows-icon-from-ico=logo.ico` |
| `--include-data-files=<源路径>=<目标路径>` | 打包外部静态资源（如 UI、图片、模型等） | `--include-data-files=main.ui=main.ui` |
| `--include-data-dir=<源目录>=<目标目录>` | 打包整个静态资源文件夹 | `--include-data-dir=assets=assets` |
| `--output-dir=<目录名>` | 指定编译产物的输出路径（避免污染当前根目录） | `--output-dir=dist` |
| `--show-progress` | 在终端实时展示编译百分比与当前处理模块 | `--show-progress` |
| `--remove-output` | 编译成功后自动删除巨大的临时中间编译文件夹（`.build` 目录） | `--remove-output` |

---

## 四、PySide6 / PyQt 项目标准打包范例

### 1. 开发排查与分发（推荐：独立文件夹模式）

```powershell
nuitka --standalone \
  --enable-plugin=pyside6 \
  --windows-console-mode=disable \
  --include-data-files=main.ui=main.ui \
  --show-progress \
  --output-dir=dist \
  main.py

```

* **结果**：编译产物存放在 `dist/main.dist/` 中，把整个 `main.dist` 文件夹压缩分发给用户即可。

### 2. 便携分发（单个 EXE 模式）

```powershell
nuitka --onefile \
  --enable-plugin=pyside6 \
  --windows-console-mode=disable \
  --include-data-files=main.ui=main.ui \
  --windows-icon-from-ico=icon.ico \
  --remove-output \
  --output-dir=dist \
  main.py

```

---

## 五、代码中文件路径兼容规范

无论是 PyInstaller 还是 Nuitka，单文件模式运行时外部资源都会解压到特定的目录。为了让外部文件（如 `.ui`、图片、音频等）在**源码直接运行**与**打包后运行**都能准确定位，请在 Python 代码中使用如下通用路径写法：

```python
import os
import sys


def get_resource_path(relative_path: str) -> str:
  """获取资源的绝对路径（兼容开发环境、Nuitka 与 PyInstaller）"""
  if getattr(sys, 'frozen', False):
    # 打包运行环境
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
  else:
    # 源码本地运行环境
    base_path = os.path.dirname(os.path.abspath(__file__))

  return os.path.join(base_path, relative_path)


# 使用示例：
ui_path = get_resource_path('main.ui')

```

---

## 六、常见问题与注意事项

1. **交互提示下载询问**：
* 提示下载 `Dependency Walker` 时：输入 `Yes`，用于分析系统级 DLL 依赖。
* 提示下载 `ccache` 时：输入 `Yes`，用于加速后续的重复编译。
* 提示下载 `gcc (MinGW64)` 时：输入 `Yes`，自动配置所需的 C 语言编译器。


2. **杀毒软件误报 / 编译卡死**：
* Nuitka 会在短时间内高频调用编译器生成并删除大量中间二进制文件，极易被 Windows Defender 实时监控拦截导致编译巨慢甚至卡死。建议在编译前将项目所在文件夹加入 Windows Defender 的“排除项（白名单）”。


3. **中文路径与名称**：
* 入口 Python 脚本名、项目工作路径尽可能使用纯英文字符，避免 C 编译器在解析路径时出现编码错误。