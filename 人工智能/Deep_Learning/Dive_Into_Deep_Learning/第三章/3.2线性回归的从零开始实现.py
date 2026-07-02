# ==================== 导入模块 ====================
import torch
import random
import matplotlib.pyplot as plt
"""
# ==================== 3.2.1 生成数据集 ====================
# 根据有噪声的线性模型构造一个人造数据集，需要使用有限样本数据集恢复模型参数
# y=Xw+b+ϵ
def synthetic_data(w, b, num_examples): #@save
    '''生成y=Xw+b+噪声'''
    X = torch.normal(0, 1, (num_examples, len(w))) 
    # 生成特征矩阵X：num_examples个样本，每个样本len(w)个特征
    
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('feature:', features[0], '\nlabel:', labels[0])

# 绘制 features 的第2列（索引1）与 labels 的散点图
plt.figure(figsize=(8, 5))  # 设置图表大小
plt.scatter(
    features[:, 1].detach().numpy(),  # 特征第二列（转换为numpy数组）
    labels.detach().numpy(),          # 标签（转换为numpy数组）
    s=1,                              # 点的大小
    alpha=0.6                         # 透明度（避免点重叠）
)
plt.xlabel('Feature 2 (x2)', fontsize=12)  # x轴标签
plt.ylabel('Label (y)', fontsize=12)       # y轴标签
plt.title('Scatter Plot: Feature 2 vs Label', fontsize=14)  # 标题
plt.grid(alpha=0.3)  # 添加网格（增强可读性）
plt.show()  # 显示图表

# ==================== 3.2.2 读取数据集 ====================
def data_iter(batch_size, features, labels):
# 定义一个生成器函数 通过yield返回数据
    num_examples = len(features) # 特征数据长度为样本总数
    indices = list(range(num_examples)) # 生成连续0-num_examples-1的连续整数列表
    random.shuffle(indices) # 洗牌
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor( 
            indices[i: min(i+batch_size, num_examples)])
        # torch.tensor()将索引列表转成Python张量
        # min(i+batch_size, num_examples)：处理最后一批数据边界情况
        # （如num_example=95, i=90，batch_size=10，最后一批i+batch_size=100超过95，min取95）
        # indices[i:i+batch_size]：截取当前批次索引
        yield features[batch_indices], labels[batch_indices]
        # 生成并返回当前批次的特征+标签
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# ==================== 3.2.3 初始化模型参数 ====================
w = torch.normal(0, 0.01, size=(2, 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)
# ==================== 3.2.4 定义模型 ====================
def linreg(X, w, b): #@save
    '''线性回归模型'''
    return torch.matmul(X, w) + b
# ==================== 3.2.5 定义损失函数 ====================
def squared_loss(y_hat, y): #@save
    '''均方损失'''
    return (y_hat - y.reshape(y_hat.shape)) ** 2/2
# ==================== 3.2.6 定义优化算法 ====================
def sgd(params, lr, batch_size): #@save
    '''小批量随机梯度下降'''
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
# ==================== 3.2.7 训练 ====================
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # X和y的小批量损失
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    
print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b - b}')
"""



'''数据准备'''
'''训练配置'''
'''迭代训练'''
'''结果验证'''

'''数据准备'''
def synthetic_data(w, b, num_examples): #@save 深度学习框架d2l的专属标记，作用是将这个函数保存到d2l的工具库中
    '''生成y=Xw+b噪声'''
    X = torch.normal(0, 1, (num_examples, len(w))) # 生成服从标准正态分布的特征矩阵X
    # torch.normal(mean, std, size) 生成指定均值、标准差形态的正态分布随机数
    # size=(num_examples, len(w)) 生成num_examples个样本，依靠权重w的数量生成特征数量
    y = torch.matmul(X, w) + b
    # torch.mutmul(X,w)使矩阵Xw相乘，再利用广播机制相加
    y += torch.normal(0, 0.01, y.shape)
    # 生成均值为0、标准差为0.01的正态分布噪声
    # size=y.shape，形状（样本和特征数）和y一样
    return X, y.reshape((-1, 1)) # 返回矩阵X和标签向量y
    # y.reshape((-1, 1))：强行将y改成N行一列的列向量，-1自动算行数，1改成1列

true_w = torch.tensor([2, -3.4]) # 真实权重设置为
true_b = 4.2 # 真实偏置
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', labels[0])

def data_iter(batch_size, features, labels):
    # 接收变量batch_size，特征矩阵（1000，2），标签列向量（1000，1）
    num_examples = len(features) # 对于多维张量，len()直接返回第一维长度
    indices = list(range(num_examples)) # 给每个样本分配编号（索引）
    random.shuffle(indices) # 打乱
    for i in range(0, num_examples, batch_size): # 按batch_size为步长，遍历所有索引
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size, num_examples)]) # 获取当前批的10个样本索引，并转成张量
        # min(i+batch_size, num_examples)：处理最后一批数据边界情况
        # （如num_example=95, i=90，batch_size=10，最后一批i+batch_size=100超过95，min取95）
        yield features[batch_indices], labels[batch_indices]
        # 返回当前批的特征 + 标签，供模型训练。
        # yield是生成器，每次循环只返回 1 批数据，用完再取下一批
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

w = torch.normal(0, 0.01, size=(2,1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

def linreg(X, w, b): 
    '''线性回归模型'''
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    '''均方损失'''
    return (y_hat - y.reshape(y_hat.shape)) **2 /2 

def sgd(params, lr, batch_size):
    '''小批量随机梯度下降'''
    # 接收w、b、lr、batch_size，根据误差梯度更新参数
    with torch.no_grad(): # 暂时关闭pytorch的梯度计算功能
        for param in params: # 逐个更新params里的参数 1.param=w 2.param=b
            param -= lr * param.grad / batch_size # 新参数=旧参数-学习率×（梯度÷批次大小）
            param.grad.zero_() # 把参数的梯度设置为0 避免下一批数据的梯度和当前批叠加

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs): # 控制训练批次为3
    for X, y in data_iter(batch_size, features, labels): # 调用函数，打乱后分批取数据
        l = loss(net(X, w, b), y) # 得到预测值y_hat → net(X,w,b)，计算loss → 预测值与真实值之差的平方损失
        l.sum().backward() # 用批次总损失 计算 反向传播，自动计算梯度
        sgd([w, b], lr, batch_size)
    with torch.no_grad(): # 关闭梯度计算
        train_l = loss(net(features, w, b), labels) # 计算所有样本的损失
        print(f'epoch{epoch + 1}, loss {float(train_l.mean()):f}')