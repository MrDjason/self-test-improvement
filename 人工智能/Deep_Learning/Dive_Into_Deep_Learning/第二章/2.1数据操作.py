# ==================== 导入模块 ====================
import torch
# ==================== 2.1.1 入门 ====================
# tensor 张量 一个轴张量为 vector 向量， 两个轴 matrix 矩阵
x = torch.arange(12)
print(x)            # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
print(x.shape)      # torch.Size([12])
print(x.numel())    # 12
X = x.reshape(3, 4) # 把张量x从形状为（12,）的行向量转换为形状为（3,4）的矩阵
print(X)            # 三行四列矩阵
print(torch.zeros((2, 3, 4))) # 生成两个三行四列矩阵
print(torch.randn(3, 4)) # 从某个特定的概率分布中随机采样来得到张量中每个元素的值
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])) # 提供列表赋予张量元素值

# ==================== 2.1.2 运算符 ====================
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x- y, x * y, x / y, x ** y)
'''
(tensor([ 3., 4., 6., 10.]),
 tensor([-1., 0., 2., 6.]),
 tensor([ 2., 4., 8., 16.]),
 tensor([0.5000, 1.0000, 2.0000, 4.0000]),
 tensor([ 1., 4., 16., 64.]))
'''
print(torch.exp(x)) # 对张量每个元素求自然指数 
#  tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03])

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))      # 生成0到11共12个连续数,数据类型是float32
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]) # 手动生成一个二维张量
print(torch.cat((X, Y), dim=0)) # dim=0 沿着行方向拼接
print(torch.cat((X, Y), dim=1)) # dim=1 沿着列方向拼接
# 将两个张量X和Y沿着指定维度dim=0/1拼接在一起
print(X.sum()) # X所有元素相加

# ==================== 2.1.3 广播机制 ====================
# 让形状不同但可兼容的张量，通过【逻辑扩展】变成同形状，从而执行按元素操作
a = torch.arange(3).reshape((3, 1)) # arange(3)生成 0、1、2三个张量元素，reshape为 (3行, 1列)
b = torch.arange(2).reshape((1, 2))
print(a, b, sep='\n') # print(a,'\n,b)会导致b前面有空行，因为sep=' '没有改变多参数分隔符
'''
此时的a、b矩阵如下，相加形状不匹配。将两个矩阵广播为一个更大矩阵
tensor([[0],
        [1],
        [2]])
tensor([[0, 1]])
'''
print(a+b)
'''
a+b=c c[i][j] = a[i][0] + b[0][j] 【a只有一列，b只有一行，所以索引只取0】
tensor([[0, 1],
        [1, 2],
        [2, 3]])
'''
# ==================== 2.1.4 索引和切片 ====================
print(X[-1], X[1:3], sep='\n')
X[0:2, :] = 12 # [0-2)行，: 代表所有列元素 
print(X)
# ==================== 2.1.5 节省内存 ====================
# 运行一些操作可能会导致为新结果分配内存
# 首先，我们不想总是不必要地分配内存
# 其次，如果我们不原地更新，其他引用仍然会指向旧的内存位置，这样我们的某些代码可能会无意中引用旧的参数
before = id(Y) # id 返回对象的内存地址标识
print(before)
Y=Y+X
print(Y)
print(id(Y) == before)

Z = torch.zeros_like(Y) # 创建一个和Y形状相同，填充全为0元素的矩阵Z
print('id(Z):', id(Z))
Z[:]= X+Y # Z[:]是切片赋值，先计算X+Y，再把零时结果写入Z已经占用的内存块
print('Z[:]= X+Y id(Z):', id(Z))
Z = X+Y
print('Z = X+Y id(Z):', id(Z))

before = id(X)
print('id(X):',before)
X += Y
print('id(X)=',id(X),'id(X)=before?:',id(X) == before)
# 后续计算中没有重复使用X，用X[:] = X + Y或X += Y来减少操作的内存开销
# ==================== 2.1.6 转换为其他Python对象 ====================
# 将深度学习框架定义的张量转换为NumPy张量，torch张量和numpy数组将共享它们的底层内存，就地操作更改一个张量也会同时更改另一个张量
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))

a = torch.tensor([3.5])  # 创建一个仅含一个元素的张量，值为3.5
print(a, a.item(), float(a), int(a))  # 分别输出张量本身、转标量的三种方式结果