# ==================== 导入模块 ====================
import torch
# ==================== 6.6.1 生成训练数据 ====================
# 确保CUDA可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 生成数据集
inputs = torch.rand(100, 3) # 随机生成shape为(100, 3)的tensor，里面每个元素为0-1
weights = torch.tensor([[1.1], [2.2], [3.3]]) # 预设的权重
bias = torch.tensor(4.4) # 预设的bias
targets = inputs @ weights + bias + 0.1*torch.randn(100, 1) # 增加一些误差
# inputs @ weights 由pytorch中矩阵乘法构成的样本特征与权重的线性组合
# + bias 调整线性模型的偏置
# + 0.1*torch.randn(100,1)模拟真实数据噪声，生成100个服从标准正态分布的随机数
print('inputs:',inputs,'weights:',weights,'bias:',bias,'targets:',targets,sep='\n')

# ==================== 6.6.2 初始化线性回归的参数 ====================
w = torch.rand((3, 1), requires_grad=True, device=device)
b = torch.rand((1,), requires_grad=True, device=device)
# 创建模型要学习的两个核心参数，并初始化他们的值

# ==================== 6.6.3 进行训练 ====================
# 将数据移至相同设备
inputs = inputs.to(device)
targets = targets.to(device)

# 设置超参数
epoch = 10000 # 训练轮次
lr = 0.003 # 学习率

for i in range(epoch):
    outputs = inputs @ w + b # 使用当前w和b，计算模型的预测值outputs
    loss = torch.mean(torch.square(outputs - targets)) # 利用MSE(所有样本的平均平方误差)算损失函数
    print('loss:', loss.item())

    loss.backward() # 自动计算loss对w和b的梯度

    # 更新参数
    with torch.no_grad(): # 关闭梯度追踪
        # 用梯度调整w和b的值，让参数更接近真实值
        w -= lr * w.grad 
        b -= lr * b.grad

    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()

print('训练后的权重 w:', w)
print('训练后的偏置 b:', b)

