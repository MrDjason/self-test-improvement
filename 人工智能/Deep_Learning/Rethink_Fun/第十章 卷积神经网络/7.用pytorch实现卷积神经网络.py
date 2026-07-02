# ==================== 导入模块 ====================
import os
import torch
import random
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from PIL import Image
# ==================== 数据准备 ====================
def verify_images(image_folder):
    classes = ['Cat', 'Dog']
    class_to_idx = {'Cat':0, 'Dog':1}
    samples = []
    for cls_name in classes:
        cls_dir = os.path.join(image_folder, cls_name)
        for fname in os.listdir(cls_dir):
        # os.listdir(cls_dir)列出目录下所有的文件名  之后赋值给fname
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            # 跳过非图片
            path = os.path.join(cls_dir, fname)
            # 拼接路径
            try:
                with Image.open(path) as img:
                # 使用Pillow库的Image.open打开图片文件
                # with自动管理文件资源，打开后自动关闭，避免内存泄漏
                    img.verify()
                    # Pillow提供的图片验证方法，检查图片文件的格式是否有效
                samples.append((path, class_to_idx[cls_name]))
                # 通过验证则将(完整路径, 类别索引)元组添加到samples列表
            except Exception:
                print(f'Warning: Skipping corrupted image {path}')
            # 发生错误则跳过损坏图片，继续处理其他图片
    return samples
    # 返回所有验证通过的图片样本列表

class ImageDataset(Dataset):
    def __init__(self, samples, transform=None):
    # transform=None,用于传入图像预处理、增强的操作
        self.samples = samples
        self.transform = transform

    def __len__(self):
    # 必须实现的魔法方法，返回数据集的总样本数
        return len(self.samples)
    
    def __getitem__(self, idx):
    # 必须实现的魔法方法。根据索引idx返回对应的单条数据
        path, label = self.samples[idx]
        # 根据索引idx从self.samples取出对应的图像路径和标签
        with Image.open(path) as img:
        # 打开指定路径的图像文件，并通过with保证文件操作完成后自动关闭
            img = img.convert('RGB')
            # 将文件转换为RGB格式
            if self.transform:
            # 判断是否传入了图像预处理操作transform(非None时执行)
                img=self.transform(img)
                # 对PIL图像img执行transform定义的预处理操作
        return img, label

class CNNModel(nn.Module):
# 定义卷积神经网络
    def __init__(self):
        super().__init__()
        # 调用父类nn.Module的初始化方法
        # 用nn.Sequential构建一个顺序执行的层容器，容器内的层会按代码顺序依次执行
        self.model = nn.Sequential(
            # 第一组 卷积+激活+池化
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二组 卷积+激活+池化
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三组 卷积+激活+池化
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第四组 卷积+激活+池化
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 1x1卷积层
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1), # 1x1卷积
            nn.AdaptiveAvgPool2d((1, 1)), # 自适应平均池化层
            nn.Flatten(), # 展平层
            nn.Sigmoid()  # Sigmoid激活层 
        )

    def forward(self, x):
    # 定义模型的前向传播逻辑
        return self.model(x)
        # 将输入张量x传入self.model顺序层容器，按顺序执行所有层，返回最终输出

# ==================== 验证模型正确率 ====================
def evaluate(model, test_dataloader):
    model.eval()
    # 切换为评估模式
    val_correct = 0
    val_total = 0

    with torch.no_grad():
    # 禁用梯度
        for inputs, labels in test_dataloader:
        #  
            inputs = inputs.to(DEVICE)
            # 将图像张量迁移到指定计算设备
            labels = labels.float().unsqueeze(1).to(DEVICE)
            # 对标签张量做三步处理：从int转换为float、unsqueeze(1)增加一个维度、迁移设备 
            outputs = model(inputs)
            # 将批次输入传入模型，执行前向传播，得到模型输出
            preds = (outputs > 0.5).float()
            # outputs > 0.5:判断每个样本概率是否大于0.5，返回布尔张量
            # .float将布尔值转为float类型(True=1.0,Flase=0.0)，与标签类型一致
            val_correct += (preds == labels).sum().item()
            # 计算当前批次中预测正确的样本数，并累加到总正确数
            val_total += labels.size(0)
            # 累加当前批次的样本总数到全局总样本数
            # label.size(0)获取张量的第一个维度
        val_acc = val_correct / val_total
        return val_acc
    
# ==================== 加载数据 ====================
if __name__ == '__main__':
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'res', 'PetImages')
    BATCH_SIZE = 64
    IMG_SIZE = 128
    EPOCHS = 10
    LR = 0.001
    PRINT_STEP = 100

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_samples = verify_images(DATA_DIR) # 验证图像有效性
    random.seed(42) # 设置随机种子
    random.shuffle(all_samples) # 打乱所有样本
    train_size = int(len(all_samples)*0.8) #训练集占总样本的80%
    train_samples = all_samples[:train_size] # 前80%作为训练集
    valid_samples = all_samples[train_size:] # 后20%作为测试集

    data_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # 将图像缩放到目标尺寸
        transforms.ToTensor(), # 将PIL图像(0-255像素)转为Pytorch张量(0-1浮点数) 将HWC转为CHW
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 对图片归一化
    ])

    train_dataset = ImageDataset(train_samples, data_transform)
    valid_dataset = ImageDataset(valid_samples, data_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = CNNModel().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ==================== 定义训练循环 ====================
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        model.train()
        running_loss = 0.0

        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs =inputs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) # 用BCELoss比较两值，得到当前批次的损失值
            loss.backward() # 反向传播
            optimizer.step() # 按学习率LR更新模型的所有参数

            running_loss += loss.item() # 将当前批次的损失值累加到running_loss

            if (step + 1 ) % PRINT_STEP == 0:
            # 每训练PRINT_STEP=100个批次，打印一次平均损失
                avg_loss = running_loss / PRINT_STEP
                print(f' STEP [{step + 1} - Loss:{avg_loss:.4f}]')