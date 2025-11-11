# ==================== 导入模块 ====================
import torch
import matplotlib.pyplot as plt
# ==================== 3.2.1 生成数据集 ====================
# 根据有噪声的线性模型构造一个人造数据集，需要使用有限样本数据集恢复模型参数
# y=Xw+b+ϵ
def synthetic_data(w, b, num_examples): #@save
    '''生成y=Xw+b+噪声'''
    X = torch.normal(0, 1, (num_examples, len(w))) 
    # 生成特征矩阵X：num_examples个样本，每个样本len(w)个特征
    
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('feature:', features[0], '\nlabel:', labels[0])

# 绘制 features 的第2列（索引1）与 labels 的散点图
plt.figure(figsize=(8, 5))  # 设置图表大小
plt.scatter(
    features[:, 1].detach().numpy(),  # 特征第二列（转换为numpy数组）
    labels.detach().numpy(),          # 标签（转换为numpy数组）
    s=1,                              # 点的大小
    alpha=0.6                         # 透明度（避免点重叠）
)
plt.xlabel('Feature 2 (x2)', fontsize=12)  # x轴标签
plt.ylabel('Label (y)', fontsize=12)       # y轴标签
plt.title('Scatter Plot: Feature 2 vs Label', fontsize=14)  # 标题
plt.grid(alpha=0.3)  # 添加网格（增强可读性）
plt.show()  # 显示图表
# ==================== 3.2.2 读取数据集 ====================
