# ==================== 导入模块 ====================
import torch
import os
import pandas as pd
# ==================== 2.2.1 读取数据集 ====================
data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res', 'house_tiny.csv')
os.makedirs(os.path.dirname(data_file), exist_ok=True)
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # 列名
    f.write('NA,Pave,127500\n') # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
data = pd.read_csv(data_file)
print(data)
# ==================== 2.2.2 处理缺失值 ====================
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True)) # numeric_only=True 只对数值进行计算
# data.iloc[:, 0:2] 取所有行+第0、1列
# inputs.fillna(inputs.mean()) 用数值型列平均值填充inputs中的缺失值
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na = True) # 对分类列做独热编码,dummy_na=True:把NA当成独立分类
print(inputs)
# pd.get_dummies()把一个分类列拆分多个0/1项，每一个列对应一个分类
# ==================== 2.2.3 转换为张量格式 ====================
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(X, y, sep='\n')