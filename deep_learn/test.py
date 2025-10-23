import torch
from torch.utils.tensorboard import SummaryWriter

# 确保CUDA可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成数据
inputs = torch.rand(100, 3)  
weights = torch.tensor([[1.1], [2.2], [3.3]])
bias = torch.tensor(4.4)
targets = inputs @ weights + bias + 0.1 * torch.randn(100, 1)  

# 创建SummaryWriter实例
writer = SummaryWriter(log_dir=r"C:\Users\13218\anaconda_logdir")

# 初始化参数
w = torch.rand(3, 1, requires_grad=True, device=device)
b = torch.rand(1, requires_grad=True, device=device)

# 数据移至设备
inputs = inputs.to(device)
targets = targets.to(device)

epoch = 10000
lr = 0.003

for i in range(epoch):
    outputs = inputs @ w + b
    loss = torch.mean(torch.square(outputs - targets))
    print(f"epoch {i}, loss: {loss.item()}")  # 加epoch编号，方便观察进度
    writer.add_scalar("loss/train", loss.item(), i)  # 记录日志
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()

# 关键：关闭writer，确保日志写入磁盘
writer.close()  

print("训练后的权重 w:", w)
print("训练后的偏置 b:", b)