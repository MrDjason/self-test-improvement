# ==================== 导入模块 ====================
import time
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
# ==================== 3.1.2 矢量化加速 ====================
n = 10000           # 定义张量长度
a = torch.ones([n]) # 创建张量a、b，形状为[n]（一维数组），所有元素均为1
b = torch.ones([n])

class Timer: #@save
    # @save 是自定义装饰器，用于保存类、函数
    '''记录多次运行时间'''
    def __init__(self):
        self.times=[] # 初始化时间列表
        self.start()  # 调用start方法，开始计时
    def start(self):
        '''启动计时器'''
        self.tik = time.time() # time.time()获取当前秒数，存入self.tik
    def stop(self):
        '''停止计时器并将时间记录在列表中'''
        self.times.append(time.time() - self.tik) # 计算并记录耗时
        return self.times[-1]
    def avg(self):
        '''返回平均时间'''
        return sum(self.times) / len(self.times)
    def cumsum(self):
        '''返回累计时间'''
        # 将times列表转numpy数组→调用cumsum() 
        return np.array(self.times).cumsum().tolist()

c = torch.zeros(n) # 创建一个全为0的长度为10000的一维张量c
timer = Timer()    # 创建Time对象，自动调用__init__，记录当前时间点
for i in range(n): # 循环10000次
    c[i] = a[i] + b[i] # 拟定一个计算记录耗时
print(f'{timer.stop():.5f} sec')

timer.start() # 重启计时器
d = a + b     # 直接进行整体加法
print(f'{timer.stop():.5f} sec')

# ==================== 3.1.3 正态分布与平方损失 ====================
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x-mu)**2)

# 使用numpy进行可视化
x = np.arange(-7, 7, 0.01)

# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]

# 用matplotlib可视化
plt.figure(figsize=(4.5, 2.5)) # 设置图像大小

for mu, sigma in params:
    plt.plot(x, normal(x, mu, sigma), label=f'mean {mu}, std {sigma}') #绘制x-y曲线，设置图例标签

    # 设置坐标轴标签
    plt.xlabel('x')
    plt.ylabel('y=f(x)')
    # 显示图例
    plt.legend()
    # 显示图片
    plt.show()
