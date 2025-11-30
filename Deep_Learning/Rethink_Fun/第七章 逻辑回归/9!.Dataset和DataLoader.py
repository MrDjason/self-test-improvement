# Pytorch Dataset与DataLoader
# ==================== 导入模块 ====================
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import os
# ==================== 7.9.1 Dataset ====================
# 创造一个符合Pytorch规范的自定义数据集类
# 任何数据集都需要继承torch.utils.data.Dataset并实现两个方法：__len__ 和 __getitem__(idex)
# __len__ 需要返回整个数据集样本的个数
# __getitem__ 需要根据样本的index返回具体的样本
class TitanicDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        # 定义数值特征的均值字典，用于后续数据标准化（提前计算好的固定值）
        self.mean = {
            'Pclass': 2.236695,
            'Age': 29.699118,
            'SibSp': 0.512605,
            'Parch': 0.431373,
            'Fare': 34.694514,
            'Sex_female': 0.365546,
            'Sex_male': 0.634454,
            'Embarked_C': 0.182073,
            'Embarked_Q': 0.039216,
            'Embarked_S': 0.775910
        }
        # 定义数值特征的标准差字典，与均值配合实现标准化
        self.std = {
            'Pclass': 0.838250,
            'Age': 14.526497,
            'SibSp': 0.929783,
            'Parch': 0.853289,
            'Fare': 52.918930,
            'Sex_female': 0.481921,
            'Sex_male': 0.481921,
            'Embarked_C': 0.386175,
            'Embarked_Q': 0.194244,
            'Embarked_S': 0.417274
        }
        self.data = self._load_data() # 调用_load_data完成数据加载和预处理，存储df
        self.feature_size = len(self.data.columns) - 1

    def _load_data(self):
        df = pd.read_csv(self.file_path) # 用pandas读取CSV文件，存志DataFrame对象df
        df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
        df = df.dropna(subset=['Age']) # 删除Age有缺失的行
        df = pd.get_dummies(df, columns=['Sex', 'Embarked'], dtype=int) # 两类拆成两列进行one-hot编码

        # 进行数据的标准化
        base_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        # 循环对每个数值特征做标准化:(原始值 - 均值) / 标准差，消除量纲影响
        for i in range(len(base_features)):
            df[base_features[i]] = (df[base_features[i]] - self.mean[base_features[i]]) /self.std[base_features[i]]
        return df
    
    def __len__(self):
        return len(self.data) # 直接返回处理后的DataFrame的行数，即样本总数
    
    def __getitem__(self, idx): # 按索引idx返回单个样本
        # 提取索引为idx的特征:删除标签列Survived，取对应行的数值，转为数组
        features =  self.data.drop(columns = ['Survived']).iloc[idx].values
        label = self.data['Survived'].iloc[idx] # 提取索引为idx的标签（Survived列的值，0/1）
        # 转为PyTorch张量，返回特征和标签
        return torch.tensor(features, dtype=torch.float32),torch.tensor(label, dtype=torch.float32)
# ==================== 7.9.2 DataLoader ====================
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'res', 'train.csv')
dataset = TitanicDataset(file_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for inputs, labels in dataloader:
    print(inputs.shape, labels.shape)
    break