# ==================== 导入模块 ====================
import torch
# ==================== 6.5.1 简单的例子 ===================
# z=log(3x+4y)^2
# ∂z/∂x ≈ 0.8571 
# ∂z/∂y ≈ 1.1429
# ==================== 6.5.2 PyTorch里的自动求梯度 ====================
x = torch.tensor(1.0, requires_grad=True) # 创建值为1的标量x，并开启梯度追踪
y = torch.tensor(1.0, requires_grad=True) # 创建值为1的标量y，并开启梯度追踪
v = 3*x + 4*y # 因为x、y的requires_grad=True，所以v会自动继承
u = torch.square(v) # 等价于v**2，同样u会自动继承
z = torch.log(u) 

z.backward() # 反向传播求梯度 backward()会自动计算梯度，并将结果存在.grad属性

print('x grad:', x.grad) # x grad: tensor(0.8571)
print('y grad:', y.grad) # y grad: tensor(1.1429)
