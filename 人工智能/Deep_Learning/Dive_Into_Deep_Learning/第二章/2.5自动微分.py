# ==================== 导入模块 ====================
import torch
# ==================== 2.5.1 简单例子 ====================
x = torch.arange(4.0)
print(x) #  tensor([0., 1., 2., 3.])
# 需要一个地方来存储梯度,值得注意的是，不会在每次对一个参数求导时都分配新的内存
x.requires_grad_(True) # 等价于x=torch.arange(4.0, requires_grad=True)
# 告诉 PyTorch需要追踪该张量的计算过程，以便后续求导
print(x.grad) # x.grad：用于存储y关于x的梯度的属性，默认None
y = 2 * torch.dot(x, x) # 结算结果得到标量（原来是向量）
'''y=2*(x₀²+x₁²+x₂²+x₃²)'''
print(y) #  tensor(28., grad_fn=<MulBackward0>)

y.backward()  # 自动求导(反向传播)
'''
正向是计算的顺序，反向是求梯度的顺序
正向传播：x（参数：[0.,1.,2.,3.]） → 第一步计算 s = x₀²+x₁²+x₂²+x₃²（中间结果） → 第二步计算 y = 2*s（最终输出：28.）
反向传播：
1.先算 y 对最外层中间结果 s 的梯度：∂y/∂s = 2（因为 y=2s，导数是 2）；
2.再算 s 对 x 的梯度：∂s/∂xᵢ = 2xᵢ（因为 s=x₀²+x₁²+...，对每个 xᵢ的导数是 2xᵢ）；
3.最后用链式法则合并：∂y/∂xᵢ = ∂y/∂s * ∂s/∂xᵢ = 2*2xᵢ = 4xᵢ（这就是最终 x 的梯度）
'''

# 计算 y 关于 x 每个元素的梯度（∂y/∂x₀, ∂y/∂x₁, ∂y/∂x₂, ∂y/∂x₃）
'''
∂y/∂x₀ = 2*(2x₀) = 4x₀ → 4*0=0
∂y/∂x₁ = 4x₁ → 4*1=4
∂y/∂x₂ = 4x₂ → 4*2=8
∂y/∂x₃ = 4x₃ → 4*3=12
'''
print(x.grad) #  tensor([ 0., 4., 8., 12.])

print(x.grad == 4*x) #  tensor([True, True, True, True])
x.grad.zero_() # 梯度清零
y = x.sum()    # 重新定义y：y = x₀ + x₁ + x₂ + x₃（求和）
y.backward()   # 计算梯度
print(x.grad)  # tensor([1., 1., 1., 1.])

# ==================== 2.5.2 非标量变量的反向传播 ====================
# 当y不是标量，向量y关于向量x的导数为一个矩阵。高阶和高维y、x，求导结果是一个高阶张量
x.grad.zero_()     # 梯度清零 
y = x*x            # x = torch.arange(4.0)，得到向量输出 
y.sum().backward() # 将y求和，自动算梯度并储存到grad中
# pytorch 只支持对标量求和，所以将y求和后再反向传播
print(x.grad)      # tensor([0., 2., 4., 6.])

# ==================== 2.5.3 分离计算 ====================
# 固定部分计算（比如预训练模型的输出），只让梯度更新另一部分参数；
# 避免不必要的梯度计算（节省内存 / 算力）；
# 解决梯度回传混乱
x = torch.tensor([1., 2., 3., 4.], requires_grad=True)
y = x * x  # 计算图：x → y（y依赖x）
u = y.detach()  # 分离y，u和y数值相同，但梯度不回传x
z = u * x  # 计算图：x → z（u是常数，不参与梯度回传）
z.sum().backward()  # 求z对x的梯度

print(x.grad)  # 结果是u（即[1.,4.,9.,16.]），而非3x²
# ==================== 2.5.4 Python控制流的梯度计算 ====================
# PyTorch自动微分支持Python 控制流（如if、while、任意函数调用等）场景下的梯度计算 
def f(a):
    b = a * 2
    while b.norm()< 1000: # x.norm() 关于标量，为绝对值范数，关于高维张量，默认所有元素进行2范数计算
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad== d/a)