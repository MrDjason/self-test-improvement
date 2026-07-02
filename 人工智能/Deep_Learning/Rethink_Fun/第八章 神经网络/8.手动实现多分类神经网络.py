# ==================== 导入模块 ====================
import torch
from torch.utils.data import DataLoader, Dataset
import os
# ==================== 加载数据 ====================
class MNISTDataset(Dataset):
    def __init__(self, file_path):
        self.images, self.labels = self._read_file(file_path)

    def _read_file(self, file_path): # _表示私有、内部方法
        images=[] # 每一个images存储的元素是长度784的列表
        labels=[]

        with open(file_path, 'r') as f:
            next(f) # 跳过标题行
            # next()是Python内置函数，作用是读取可迭代对象的下一个元素
            # next()本质是调用迭代器的__next__()方法
            # 从当前指针位置读取，直到遇到换行符\n为止
            for line in f:
                line = line.rstrip('\n') # 去除末尾换行符
                items = line.split(',') # 按,分割每行数据，得到字符串列表
                images.append([float(x) for x in items[1:]]) # 第二列以后的数转浮点数添加到images
                labels.append(int(items[0])) # 第一列转整数添加到labels
        return images, labels
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index] # 获得图像像素列表和标签
        image = torch.tensor(image)
        image = image / 255.0 # 归一化
        # MNIST是8位灰度图，像素值范围在[0,255]，经过归一化可将像素值范围线性缩放在[0,1]之间
        image = (image - 0.1307) / 0.3081 # 标准化
        # 0.1307，均值 = 所有归一化后像素值的总和 / 像素的总数量
        # 0.3081，标准差 = 根号(方差 = 每个像素值与均值的差的平方和 / 像素的总数量)
        label = torch.tensor(label)
        return image, label
    
    def __len__(self):
        return len(self.images)
    
# ==================== 全局设置 ====================
batch_size = 64

train_dataset = MNISTDataset(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'res', 'mnist_train.csv'))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MNISTDataset(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'res', 'mnist_test.csv'))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


learning_rate = 0.1
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ==================== 定义神经网络并初始化参数 ====================
layer_sizes = [28*28, 128, 128, 128, 64, 10] # 输入 隐藏层 隐藏层 隐藏层 隐藏层 输出
# 定义全连接神经网络层级结构

# 手动初始化参数
weights = [] # 存储每一层权重矩阵
biases = []  # 存储每一层偏置向量
for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
# zip([A],[B]) 将ab打包成元组，放在新列表当中→[(A),(B)]
# 通过zip得到5组维度:(784,128)、(128,128)、(128,128)、(128,64)、(64,10)

    # 初始化权重矩阵W
    W = torch.randn(in_size, out_size, device=device) * torch.sqrt(torch.tensor(2/in_size))
    # 生成形状为(输入维度,输出维度)的矩阵，元素服从标准正态分布，并将权重放在指定设备
    # torch.sqrt()对张量开平方根
    # torch.tensor(2/in_size) 将2/in_size转为张量
    # ReLU会将所有负数置零，只保留正数，50%的数值置零，剩下正数部分的方差减半
    # 权重不做缩放，多层神经网络每一层输出方差会持续衰减，叠加几层方差趋于0梯度消失
    # torch.sqrt(torch.tensor(2/in_size))作用是生成缩放因子张量，调整torch.randn()初始化的权重矩阵的方差    
    # Var(z) = n × Var(x) × Var(W)带入Var(x)=1（标准化后）得2 = in_size × 1 × σ²
    # σ² = 2/in_size 则 σ = √(2/in_size)

    # 初始化偏置向量b
    b = torch.zeros(out_size, device=device)
    weights.append(W)
    biases.append(b)

# ==================== 函数定义 ====================
# ReLU
def relu(x):
    return torch.clamp(x, min=0)
    # torch.clamp(x, min=0)是PyTorch的截断函数，把张量所有小于0的元素强制设为0

def relu_grad(x):
    return (x>0).float()
    # Pytorch中张量比较运算符(>、<等)是按元素操作，遍历张量x的每一个元素，分别与0做大于比较
    # 最终生成一个和x形状完全相同的布尔张量
    #.float()：将布尔张量转为浮点型（True→1.0，False→0.0）。

# Softmax
def softmax(x):
    x_exp = torch.exp(x - x.max(dim=1, keepdim=True).values)
    # dim=1 沿着列维度计算最大值
    # keepdim=True 保持维度不变，结果形状为(batch_size, 1)
    # x-x.max(...) 通过广播，每行的每个元素都减去改行的最大值，确保e^x中的x肯定为负数
    # 直接计算e^x 如果x较大会超出浮点数表示范围，变成inf
    return x_exp / x_exp.sum(dim=1, keepdim=True)

# 交叉熵损失函数
def cross_entropy(pred, labels):
    N = pred.shape[0] # 获取批次样本数
    one_hot = torch.zeros_like(pred) # 创建和pred形状相同的全0张量，储存one-hot编码
    one_hot[torch.arange(N), labels] = 1 # 生成label的one-hot编码
    # torch.arange(N) 生成0到N-1的一维序列，本质是索引【0至N-1】
    # one_hot[样本索引, 类别索引] = 1 是PyTorch的高级索引
    # one_hot[torch.arange(N), labels] 将 torch.arange(N) 和 labels 按位置一一配对
    # 得到 N 个 (行索引, 列索引) 组合，然后把这些位置的元素设为 1。
    loss = - (one_hot * torch.log(pred + 1e-8)).sum() / N # 计算平均Loss
    return loss,one_hot 

# 训练循环
for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        x = images.to(device)
        y = labels.to(device)
        N = x.shape[0]

        # 前向传播
        activations = [x]
        pre_acts = []
        for W, b in zip(weights[:-1], biases[:-1]):
            z = activations[-1] @ W + b
            pre_acts.append(z)
            a = relu(z)
            activations.append(a)
        # 输出层
        z_out = activations[-1] @ weights[-1] + biases[-1]
        pre_acts.append(z_out)
        y_pred = softmax(z_out)

        # 损失
        loss, one_hot = cross_entropy(y_pred, y)
        total_loss += loss.item()

        # 反向传播
        grads_W = [None] * len(weights)
        grads_b = [None] * len(biases)

        # 输出层梯度
        dL_dz = (y_pred - one_hot) / N # [N, output]
        grads_W[-1] = activations[-1].t() @ dL_dz
        grads_b[-1] = dL_dz.sum(dim=0)

        # 隐藏层梯度
        for i in range(len(weights)-2, -1, -1):
            dL_dz = dL_dz @ weights[i+1].t() * relu_grad(pre_acts[i])
            grads_W[i] = activations[i].t() @ dL_dz
            grads_b[i] = dL_dz.sum(dim=0)

        # 更新参数
        with torch.no_grad():
            for i in range(len(weights)):
                weights[i] -= learning_rate * grads_W[i]
                biases[i] -= learning_rate * grads_b[i]

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss:{avg_loss:.4f}')

# 测试集评估
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader: 
        x = images.view(-1, layer_sizes[0]).to(device)
        y = labels.to(device)
        a = x
        for W, b in zip(weights[:-1], biases[:-1]):
            a = relu(a @ W + b)
        logits = a @ weights[-1] + biases[-1]
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    print(f'Test Accuracy:{correct/total*100:.2f}%')