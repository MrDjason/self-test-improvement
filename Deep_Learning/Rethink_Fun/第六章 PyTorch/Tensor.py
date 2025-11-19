# ==================== 导入模块 ====================
import torch
import numpy as np
# ==================== 备    注 ====================
# Tensor是PyTorch对多维数组的表示
# 标量（0维）torch.tensor(3.14)
# 向量（1维）torch.tensor([1,2,3])
# 矩阵（2维）torch.tensor([1,2],[3,4])
# 高维张量 torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
# ==================== 6.4.2 创建一个Tensor ====================
# 1D Tensor
t1 = torch.tensor([1,2,3])
print(t1)

# 2D Tensor
t2 = torch.tensor([[1,2,3],[4,5,6]])
print(t2)

# 3D
t3 = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
print(t3) 

# 从 NumPy 创建 Tensor
arr = np.array([1,2,3])
t_np = torch.tensor(arr)
print(t_np)

# 创建tensor时 PyTorch会根据传入数据，自动推断tensor的类型，或者指定类型

t_1 = torch.tensor((2,2), dtype=torch.float32)
print(t_1)

# Pytorch 里的数据类型
# 整数型 torch.uint8、torch.int32、torch.int64。其中torch.int64为默认的整数类型。
# 浮点型 torch.float16、torch.bfloat16、 torch.float32、torch.float64，其中torch.float32为默认的浮点数据类型。
# 布尔型 torch.bool


# Bool类型tensor
x = torch.tensor([1,2,3,4,5])
mask = x > 2 # 生成一个布尔掩码
print(mask)  # tensor([False, False, True, True, True])

# 用布尔掩码筛选出大于2的值
filtered_x = x[mask]
print(filtered_x) # tensor([3,4,5])

# 用布尔掩码选出大于2的值，并赋值为0
x[mask]=0
print(x) # tensor([1, 2, 0, 0, 0])

# 创建一个GPU/显存的tensor
# t_gpu = torch.tensor([1,2,3],device='cuda')

# 创建一个用指定值或随机值填充的tensor
shape=(2,3)                        # 生成的tensor是2X3的二维数组 
rand_tensor = torch.rand(shape)    # 从[0,1)均匀抽样生成元素填充shape形状张量
print('rand_tensor:',rand_tensor)   
randn_tensor = torch.randn(shape)  # 从标准正态分布抽样生成元素填充shape形状张量
print('randn_tensor:',randn_tensor)# 大部分元素集中在[-1,1]
ones_tensor = torch.ones(shape)    # 生成全为1的元素填充shape形状张量
zeros_tensor = torch.zeros(shape)  # 生成全为0的元素填充shape形状张量
twos_tensor = torch.full(shape, 2) # 生成全为2的元素填充shape形状张量
# ==================== 6.4.3 Tensor的属性 ====================
tensor = torch.rand(3,4)

print(f'Shape of tensor:{tensor.shape}')
print(f'DataType of tensor:{tensor.dtype}')
print(f'Device tensor is stored on:{tensor.device}')
# ==================== 6.4.4 Tensor的操作 ====================
X = torch.randn(4,4) # 生成一个形状为4X4的随机矩阵
print(X)
X = X.reshape(2,8)   # 通过reshape将4X4的矩阵改改变为2X8的矩阵
print(X)

X = torch.tensor([[1, 2, 3], [4, 5, 6]]) # 行优先存储
X_reshape = X.reshape(3,2) # 变换三行两列矩阵，则按 12 34 56进行排列
X_transpose = X.permute(1,0) # 交换第0个和第1个维度。二维则是行列互换，进行转置
print('reshape:',X_reshape)
print('permute:',X_transpose)

X = torch.tensor([[1,2,3],[4,5,6]])
# 扩展第0维
X_0 = X.unsqueeze(0)
print(X_0.shape, X_0) # X_0.shape 告诉每个维度的元素个数，由外及里
# 扩展第1维
X_1 = X.unsqueeze(1)
print(X_1.shape, X_1)
# 扩展第2维
X_2 = X.unsqueeze(2)
print(X_2.shape, X_2)

# 可使用tensor的squeeze方法缩减tensor大小为1的维度
x = torch.ones((1,1,3))
print(x.shape, x)
y = x.squeeze(dim=0) # 只删除第0维(且该维度大小必须为1)
print(y.shape, y)
z = x.squeeze() # 删除张量中所有大小1为的维度
print(z.shape, z)

# 数学计算
a = torch.ones((2,3))
b = torch.ones((2,3))
print(a + b)     # 加法
print(a - b)     # 减法
print(a * b)     # 逐元素乘法
print(a / b)     # 逐元素除法
print(a @ b.t()) # 矩阵乘法

# 统计函数
# tensor.sum()求和
# tensor.mean()求均值
# tensor.std()求标准差
# tensor.min()求最小值

t = torch.tensor([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]])
mean = t.mean() # 计算张量中的所有元素平均值，返回一个标量
print('mean:', mean)

mean = t.mean(dim=0) # 延第0维（行）计算，对每一列求平均值，返回一个1维张量
print('mean on dim 0:', mean) # shape为(2,),一维张量

mean = t.mean(dim=0, keepdim=True) # keepdim=True 对某一维度统计保持原来维度不变
print('keepdim:', mean) # shape为(1,2)
print('keepdim:', t.mean(dim=1)) # shape为(3,),一维张量
print('keepdim:', t.mean(dim=1,keepdim=True)) # shape为(1,3)

# 索引和切片
x = torch.tensor([[1,2,3],[4,5,6]])
print(x)
print(x[0,1])  # 方位第一行第二个元素
print(x[:,1])  # 访问第二列
print(x[1,:])  # 访问第二行
print(x[:,:2]) # 访问前两列

# 广播机制
t1 = torch.tensor((3,2))
print(t1)
t2 = t1 + 1
print(t2)

t1 = torch.tensor((3,2))
t2 = torch.tensor(2)
t3 = t1 + t2
print(t1)
print(t2)
print(t3)
'''
广播机制原则
1.维度对齐
t1 = tensor ([[1.0, 2.0],[3.0, 4.0]])

t2 = tensor ([[[10.0, 20.0],[30.0, 40.0]],
[[50.0, 60.0],[70.0, 80.0]],
[[90.0, 100.0],[110.0, 120.0]]])

t1 + t2 = tensor ([[[11.0, 22.0],[33.0, 44.0]],
[[51.0, 62.0],[73.0, 84.0]],
[[91.0, 102.0],[113.0, 124.0]]])
2.扩展维度
t1的shape为（3,1）
t2的shape为（1,4）
扩展t1的维度为（3,4）扩展t2的维度为（3,4）

最后一维的大小（2 和 3）均非 1 且不相等，无法通过广播扩展到同一维度大小
'''
# ==================== 6.4.5 利用GPU加速计算 ====================
'''
import torch
import time

# 确保 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 生成随机矩阵
size = 10000  # 矩阵大小
A_cpu = torch.rand(size, size) # 默认在CPU上创建tensor
B_cpu = torch.rand(size, size)

start_cpu = time.time()
C_cpu = torch.mm(A_cpu, B_cpu)  # 矩阵乘法
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

# 在 GPU 上计算
A_gpu = A_cpu.to(device) # 将tensor转移到GPU上
B_gpu = B_cpu.to(device)

start_gpu = time.time()
C_gpu = torch.mm(A_gpu, B_gpu)
torch.cuda.synchronize()  # 确保GPU计算完成
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

print(f"CPU time: {cpu_time:.6f} sec")
if torch.cuda.is_available():
    print(f"GPU time: {gpu_time:.6f} sec")
else:
    print("GPU not available, skipping GPU test.")
'''