# ==================== 导入模块 ====================
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import os
# ==================== 定义模型 ====================
class MNISTDataset(Dataset):
# 继承Dataset基类
    def __init__(self, file_path):
    # 初始化时读取文件，将图片/标签存入实例变量
        self.images, self.labels = self._read_file(file_path)
    
    def _read_file(self, file_path):
    # 读取文件
        images = []
        labels = []
        with open(file_path, 'r') as f:
            next(f) # 跳过标题行
            for line in f:
                items = line.strip().split(',') # 按,分割每行数据，得到字符串列表
                images.append([float(x) for x in items[1:]])
                labels.append(int(items[0]))
            return images, labels
        
    def __getitem__(self, index):
        image = torch.tensor(self.images[index], dtype=torch.float32).view(-1)
        image = image / 255.0 # 归一化
        image = (image - 0.1307) / 0.3081 # 标准化
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label
    
    def __len__(self):
        return len(self.images)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
    
    def forward(self, x):
        return self.model(x)
    # self.model = nn.Sequential(...) 是 PyTorch 中简化神经网络层堆叠与前向传播的核心写法
    # 前向传播时，数据会严格按定义顺序依次通过每一层，无需手动写每一层的计算逻辑。 
# ==================== 准备工作 ====================
# 参数设置
batch_size = 64
learning_rate = 0.1
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载
train_dataset = MNISTDataset(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'res', 'mnist_train.csv'))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MNISTDataset(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'res', 'mnist_test.csv'))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 模型、损失函数、优化器
model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# ==================== 训练模型 ====================
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels) # 计算loss

        optimizer.zero_grad() # 清理梯度
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

        total_loss += loss.item()
        # loss = criterion(...)得到的loss是PyTorch的Tensor对象
        # 通过loss.item()提取Tensor中的纯Python标量值

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss:{avg_loss:.4f}')

# ==================== 测试模型 ====================
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        # 求张量最大值对应的索引
        correct += (preds == labels).sum().item()
        # (preds == labels).sum()将布尔张量转换为数值并求和
        # item()提取Tensor中纯Python标量值，避免Tensor累加导致的内存浪费
        # correct += ...：将当前批次的正确数累加到全局correct中
        total += labels.size(0)
        # 累加当前批次的总样本数到total

print(f'Test Accuracy: {100 * correct / total:.2f}%')
