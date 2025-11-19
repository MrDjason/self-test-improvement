# ==================== 导入模块 ====================
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
# ==================== 3.3.1 生成数据集 ====================
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
# 生成权重、偏执、特征、标签
# ==================== 3.3.2 读取数据集 ====================
def load_array(data_arrays, batch_size, is_train=True):
    '''构造一个PyTorch数据迭代器'''
    dataset = data.TensorDataset(*data_arrays) # 把特征和标签打包成数据集
    # TensorDataset 作用是将特征张量和标签张量绑定在一起（按索引对应）
    return data.DataLoader(dataset, batch_size, shuffle=is_train) # 生成批量迭代器

batch_size = 10
data_iter = load_array((features, labels), batch_size) # 创建可迭代对象

next(iter(data_iter))
# iter(...) 接受一个可迭代对象，返回对应迭代器
# next(...) 用迭代器取下一批次 

# ==================== 3.3.3 定义模型 ====================
from torch import nn # 导入Pytorch的神经网络模块
net = nn.Sequential(nn.Linear(2, 1)) # 定义线性回归模型

# ==================== 3.3.4 初始化模型参数 ====================
net[0].weight.data.normal_(0, 0.01) # 初始化权重：均值0、标准差0.01的正态分布
net[0].bias.data.fill_(0)           # 初始化偏置：全部设为0

# ==================== 3.3.5 定义损失函数 ====================
loss = nn.MSELoss() # 定义均方误差损失

# ==================== 3.3.6 定义优化算法 ====================
trainer = torch.optim.SGD(net.parameters(), lr=0.03) # 随机梯度下降优化器

# ==================== 3.3.7 训练 ====================
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y) # 计算当前批次的损失：net(X)是模型预测值，y是真实标签
        trainer.zero_grad() # 清空上一轮的梯度（避免梯度累积）
        l.backward()        # 反向传播：计算损失对所有可学习参数的梯度
        trainer.step()      # 梯度下降：根据梯度更新模型参数（w和b）
    l = loss(net(features), labels) # 计算整个数据集的损失
    print(f'epoch {epoch + 1}, loss {l:f}') # 打印轮数和损失