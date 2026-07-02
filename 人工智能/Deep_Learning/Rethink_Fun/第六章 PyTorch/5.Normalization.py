# ==================== 导入模块 ====================
import torch
# ==================== 6.8.1 一个例子 ====================
# 送外卖时间 与 红绿灯数量 距离 关系
# time = 2 * lights + 0.01 * distance + 5
'''
time	lights	distance
19	    2	    1000
31	    3	    2000
14	    2	    500
15	    1	    800
43	    4	    3000
'''
# ==================== 6.8.2 用梯度下降训练 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

inputs = torch.tensor([[2, 1000],[3, 2000],[2, 500],[1, 800],[4, 3000]], dtype=torch.float, device=device)
labels = torch.tensor([[19],[31],[14],[15],[43]],dtype=torch.float, device=device)

w = torch.ones(2,1, requires_grad=True, device=device) # 初始化参数w
b = torch.ones(1, requires_grad=True, device=device) # 初始化参数b

epoch = 200
lr = 0.0000001

for i in range(epoch):
    outputs = inputs @ w + b # 计算模型的预测输出
    loss = torch.mean(torch.square(outputs - labels)) # MSE = 平均( (预测值outputs - 真实值labels)² )
    print('loss', loss.item()) # item()将张量转换为Python标量，方便查看
    loss.backward() # 自动计算损失函数对所有requires_grad=True参数（w和b）的梯度
    print('w.grad', w.grad.tolist()) # tolist()将张量转换为列表，方便查看梯度
    with torch.no_grad():
        w -= w.grad * lr
        b -= b.grad * lr 

    w.grad.zero_()
    b.grad.zero_()

# ==================== 6.8.3 对feature进行归一化 ====================
inputs = torch.tensor([[2,1000],[3,2000],[2,500],[1,800],[4,3000]], dtype=torch.float, device=device)
labels = torch.tensor([[19],[31],[14],[15],[43]], dtype=torch.float, device=device)

# 进行归一化
inputs = inputs / torch.tensor([4, 3000], device=device)

'''
inputs 有两个特征
第一列特征(lights)：取值 [1,2,3,4]，范围仅 0~4
第二列特征(distance)：取值 [500,800,1000,2000,3000]，范围是 0~3000
如果不做归一化，两个特征的数值差距高达数百倍.
针对输入特征尺度差异过大的问题（如案例中 “红绿灯数量” 范围 1-4，“距离” 范围 500-3000）
采用 “最大值归一化” 方法：让每个特征除以自身的最大值，将所有特征缩放到 [0,1] 区间，确保不同特征的取值范围一致。
'''

w = torch.ones(2, 1, requires_grad=True, device=device)
b = torch.ones(1, requires_grad=True, device=device)

epoch = 1000
lr = 0.5

for i in range(epoch):
    outputs = inputs @ w + b
    loss = torch.mean(torch.square(outputs - labels))
    print('loss', loss.item()) # loss.item()
    # 例如,print(loss)——tensor(0.3456, device='cuda:0', grad_fn=<MSELossBackward0>)
    # print(loss.item())——0.3456
    loss.backward()
    print('w.grad', w.grad.tolist()) # tolist()将张量转换为列表 
    # 例如,print(w.grad)——tensor([[0.123, 0.456], [0.789, 0.012]], device='cuda:0')
    # print(w.grad.tolist())——[[0.123, 0.456], [0.789, 0.012]]
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    
    w.grad.zero_()
    b.grad.zero_()

# ==================== 6.8.4 对特征进行标准化 ====================
# 标准化：对每个特征减去自己的均值，除以自己的标准差
inputs = torch.tensor([[2, 1000], [3, 2000], [2, 500], [1, 800], [4, 3000]], dtype=torch.float, device=device)
labels = torch.tensor([[19], [31], [14], [15], [43]], dtype=torch.float, device=device)

# 计算特征的均值和标准差
mean = inputs.mean(dim=0)
std = inputs.std(dim=0)

# 对特征进行标准化
inputs_norm = (inputs-mean)/std

w = torch.ones(2, 1, requires_grad=True, device=device)
b = torch.ones(1, requires_grad=True, device=device)

epoch = 1000
lr = 0.5

for i in range(epoch):
    outputs = inputs_norm @ w + b
    loss = torch.mean(torch.square(outputs - labels))
    print('loss', loss.item())
    loss.backward()
    print('w.grad', w.grad.tolist())
    with torch.no_grad():
        w -= w.grad * lr
        b -= b.grad * lr
    
    w.grad.zero_()
    b.grad.zero_()

# ==================== 6.8.5 预测时的归一化 ====================
# 训练模型时若对输入特征做了标准化，预测新数据时必须用训练集的均值和标准差来处理新数据
# 不能重新计算新数据的均值/标准差，否则会导致预测结果出错。
nputs = torch.tensor([[2, 1000], [3, 2000], [2, 500], [1, 800], [4, 3000]], dtype=torch.float, device=device)
labels = torch.tensor([[19], [31], [14], [15], [43]], dtype=torch.float, device=device)

# 计算每个特征的均值和标准差
mean = inputs.mean(dim=0) # 对每行进行均值计算，固定列
std = inputs.std(dim=0)
# 对特征进行标准化
inputs = (inputs-mean)/std

w = torch(2,1,requires_grad=True,device=device)
b = torch(1, requires_grad=True,devic=device)

epoch = 2000
lr = 0.1

for i in range(epoch):
    outputs = inputs @ w + b
    loss = torch.mean(torch.square(outputs - labels))
    print('loss', loss.item())
    loss.backward()
    print('w.grad', w.grad.tolist())
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero()
    b.grad.zero()

# 对新采集的数据进行预测
new_input = torch.tensor([[3,2500]], dtype=torch.float, device=device)
# 对于新的数据进行预测时，同样要进行标准化
new_input = (new_input-mean)/std
# 预测
predict = new_input @ w + b
# 打印预测结果
print('Predice:', predict.tolist()[0][0])