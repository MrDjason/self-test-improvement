import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F
import matplotlib.pyplot as plt

# 数据集的生成
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
iters = 10000

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

model = TwoLayerNet(10, 1)
optimizer = optimizers.SGD(lr) # 生成优化器
optimizer.setup(model)

for i in range(iters):
    y_pred = model.forward(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update() # 通过优化器更新
    
    if i % 1000 == 0:
        print(loss)

plt.scatter(x, y)

t = np.linspace(0, 1, 100).reshape(100, 1)
y_line = model.forward(t)

plt.plot(t, y_line.data, color = 'red', linewidth = 2)
plt.show()