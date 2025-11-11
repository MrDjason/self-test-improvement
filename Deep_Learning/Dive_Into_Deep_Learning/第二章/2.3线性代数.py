# ==================== 导入模块 ====================
import torch

# ==================== 2.3.1 标量 ====================
# 标量由只有一个元素的张量显示
x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x+y, x*y, x/y, x**y, sep='\n')
# ==================== 2.3.2 向量 ====================
# 向量可被视为标量值组成的列表，由一维张量表示向量
x = torch.arange(4)
print(x)
# 通过内置len()访问张量长度
print(len(x)) 
# 通过.shape属性访问向量长度
print(x.shape)
# ==================== 2.3.3 矩阵 ====================
A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T) # 转置

# 定义对称矩阵B并验证
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B) # 每个元素进行验证
print(B == B.T) 

# ==================== 2.3.4 张量 ====================
# 向量是一维张量，矩阵是二维张量
# 如A=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]，它的结构需要两个独立方向（行和列）才能完整描述
X = torch.arange(24).reshape(2, 3, 4) # 将一维张量改写为长度分别为2、3、4的三个轴的三维张量
print(X)
'''
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
可以用[[[看维数，由外及里
'''

# ==================== 2.3.5 张量算法性质 ====================
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone() # 通过分配新内存，将A的一个副本分配给B
print(f'A={A}',f'A+B={A+B}', sep='\n') # A和A+B
print(f'A*B={A*B}') # A⊙B

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X, (a * X),(a * X).shape, sep='\n')

# ==================== 2.3.6 降维 ==================== 
# 张量降维操作，通过求和、均值等聚合函数压缩张量维度
'''基本求和'''
x = torch.arange(4, dtype=torch.float32)
print(x,x.sum(),sep='\n') # (tensor([0., 1., 2., 3.]), tensor(6.))
print(A.shape, A.sum(),sep='\n')
# 指定张量沿哪一个轴来通过求和降低维度
'''指定维求和'''
A_sum_axis0 = A.sum(axis=0) # 按列开始求和
print(A_sum_axis0, A_sum_axis0.shape,sep='\n') #  (tensor([40., 45., 50., 55.]), torch.Size([4]))
A_sum_axis1 = A.sum(axis=1) # 按行开始求和
print(A_sum_axis1, A_sum_axis1.shape) # tensor([ 6., 22., 38., 54., 70.]) torch.Size([5])
print(A.sum(axis=[0, 1])) # tensor(190.) 与A.sum()相同
'''平均值'''
# 基本求平均值
print(A.mean(), A.sum()/A.numel(), sep='\n') # A.numel()计算元素个数
# 指定轴求平均
print(A.mean(axis=0), A.sum(axis=0)/A.shape[0], sep='\n') # (tensor([ 8., 9., 10., 11.]), tensor([ 8., 9., 10., 11.]))
'''非降维求和 Keepdims=True'''
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)
print(A/sum_A) # 将每个元素除以行均值
print(A.cumsum(axis=0)) # 每一列的元素为该行列元素之上元素之和
# ==================== 2.3.7 点积 ==================== 
# x^Ty
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype = torch.float32)
print(x, y, torch.dot(x, y))
print(torch.sum(x*y)) # 通过执行按元素乘法，然后进行求和来表示两个向量的点积
# ==================== 2.3.8 矩阵-向量积 ====================
# Ax=b → (aij)x → (aijxij)
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
x = torch.arange(4, dtype=torch.float32)
# 为矩阵A和向量x调用torch.mv(A,x)时，会执行矩阵‐向量积,得出Ax=b中的b
print(A.shape, x.shape, torch.mv(A, x), sep='\n')
# ==================== 2.3.9 矩阵-矩阵乘法 ====================
# AB A 5x4 B 4x3
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = torch.ones(4, 3)
print(torch.mm(A, B))
# ==================== 2.3.10 范数 ====================
'''
范数性质
1.f(αx)=|α|f(x)
2.f(x+y)≤f(x)+f(y)
3.f(x)≥0
'''
# 计算L2范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u)) # tensor(5.)

# 计算L1范数，受异常值影响较小
torch.abs(u).sum()

