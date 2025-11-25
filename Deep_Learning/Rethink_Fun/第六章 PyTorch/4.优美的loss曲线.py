# ==================== 导入模块 ====================
import torch
from torch.utils.tensorboard import SummaryWriter
# ==================== 6.7.2 修改代码 ====================
# 确保CUDA可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # torch.cuda.is_available()

# 生成数据
inputs = torch.rand(100, 3) # 生成shape为(100,3)的tensor，值在0-1间
weights = torch.tensor([[1.1],[2.2],[3.3]])
bias = torch.tensor(4.4)
targets = inputs @ weights + bias +0.1 * torch.randn(100,1)

# 创建一个SummaryWriter实例
writer = SummaryWriter(log_dir = 'C:/Users/13218/Documents/GitHub/self-test-improvement/Deep_Learning/Rethink_Fun/res')

# 初始化参数时直接放在CUDA上，并启用梯度追踪
w = torch.rand(3, 1, requires_grad=True, device=device)
b = torch.rand(1, requires_grad=True, device=device)

# 将数据移至相同设备
inputs = inputs.to(device)
targets = targets.to(device)

epoch = 10000
lr = 0.003

for i in range(epoch):
    outputs = inputs @ w + b
    loss = torch.mean(torch.square(outputs - targets))
    print('loss:', loss.item())
    writer.add_scalar('loss/train', loss.item(), i)
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    
    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()

print('训练后的权重w:',w)
print('训练后的偏置b:',b)