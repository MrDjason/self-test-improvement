# ==================== 导入模块 ====================
import numpy as np
import matplotlib.pyplot as plt
# ==================== 2.4.1 导数和微分 ====================
# 部分库不能用，稍作修改
def f(x):
    return 3*x**2-4*x
def numerical_lim(f, x, h):
    # 用有限步长h近似计算函数f在x点导数
    return (f(x+h)-f(x))/h

h=0.1
for i in range(5):
    print(f'h={h:.5f},numerical limit={numerical_lim(f, 1, h):.5f}')
    h*=0.1
'''可视化'''
# def use_svg_display(): #@save
#    backend_inline.set_matplotlib_formats('svg')

# 配置matplotlib显示格式（兼容普通Python环境，svg格式更清晰）
def use_svg_display():
    """使用svg格式显示绘图"""
    try:
        plt.rcParams['figure.format'] = 'svg'
    except:
        pass  # 非Jupyter环境忽略，不影响运行

# 设置图表大小
def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

# 设置坐标轴属性
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

# 绘制数据点和曲线
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else plt.gca()  # 使用原生matplotlib的gca()
    
    # 判断是否为单轴数据
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))
    
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

# 绘制函数 f(x) 及其在 x=1 处的切线
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
plt.show()  

# ==================== 2.4.2 偏导数 ====================
# ==================== 2.4.3 梯度 ====================