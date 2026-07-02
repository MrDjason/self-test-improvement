# ==================== 导入模块 ====================
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import os
# ==================== 7.11.1 PyTorch里的优化器 ====================
'''
# 定义优化器，传入模型的参数，并设定固定的学习率
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(epochs):
    for features, labels in DataLoader(myDataset, batch_size=256):
        optimizer.zero_grad() # 优化器清理管理参数的梯度值
        loss = forward_and_compute_loss(featuree, labels) # 前向传播并计算loss
        loss.backward() # loss反向传播，每个参数计算梯度
        optimizer.step() # 优化器进行参数更新
'''

# ==================== 7.11.2 数据集的划分 ====================
# train.csv 891条数据，前800条作为训练集，后91条作为验证集
# ==================== 7.11.3 Titanic逻辑回归代码 ====================
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x)) # Logistic Regression 输出概率

class TitanicDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.mean = {
            "Pclass": 2.236695,
            "Age": 29.699118,
            "SibSp": 0.512605,
            "Parch": 0.431373,
            "Fare": 34.694514,
            "Sex_female": 0.365546,
            "Sex_male": 0.634454,
            "Embarked_C": 0.182073,
            "Embarked_Q": 0.039216,
            "Embarked_S": 0.775910
        }

        self.std = {
            "Pclass": 0.838250,
            "Age": 14.526497,
            "SibSp": 0.929783,
            "Parch": 0.853289,
            "Fare": 52.918930,
            "Sex_female": 0.481921,
            "Sex_male": 0.481921,
            "Embarked_C": 0.386175,
            "Embarked_Q": 0.194244,
            "Embarked_S": 0.417274
        }

        self.data = self._load_data()
        self.feature_size = len(self.data.columns) - 1 # 特征数量等于csv栏数量-1

    def _load_data(self):
        df = pd.read_csv(self.file_path)
        df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin']) # 删除不用列
        df = df.dropna(subset=['Age']) # 删除Age有缺失的行
        df = pd.get_dummies(df, columns=['Sex', 'Embarked'], dtype=int) # 独热编码

        # 数据标准化
        base_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        for i in range(len(base_features)):
            df[base_features[i]] = (df[base_features[i]] - self.mean[base_features[i]]) / self.std[base_features[i]]
        return df
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.data.drop(columns=['Survived']).iloc[idx].values
        label = self.data['Survived'].iloc[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dataset = TitanicDataset(os.path.join(file_path, 'res', 'train.csv'))
validation_dataset = TitanicDataset(os.path.join(file_path, 'res', 'validation.csv'))

model = LogisticRegressionModel(train_dataset.feature_size)
model.to('cpu')
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 100

for epoch in range(epochs):
    correct = 0 # 对的数量
    step = 0 # 步数
    total_loss = 0 # 总误差
    for features, labels in DataLoader(train_dataset, batch_size=256, shuffle=True):
        step += 1
        features = features.to('cpu')
        labels = labels.to('cpu')
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        correct += torch.sum(((outputs >= 0.5) == labels))
        loss = torch.nn.functional.binary_cross_entropy(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss：{total_loss/step:.4f}')
    print(f'Training Accuracy:{correct / len(train_dataset)}')

model.eval()
with torch.no_grad():
    correct = 0
    for features, labels in DataLoader(validation_dataset, batch_size=256):
        features = features.to('cpu')
        labels = labels.to('cpu')
        outputs = model(features).squeeze()
        correct += torch.sum(((outputs >=0.5) == labels))
    print(f'Validation Accuracy: {correct / len(validation_dataset)}')

'''
model.train () 的作用：用于明确模型的工作状态（训练 / 预测）
逻辑回归模型在两种状态下表现一致,但后续会学到的批量归一化、Dropout 模块
训练和预测时的工作模式不同,届时需区分
'''
class TitanicDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.mean = {
            "Pclass": 2.236695,
            "Age": 29.699118,
            "SibSp": 0.512605,
            "Parch": 0.431373,
            "Fare": 34.694514,
            "Sex_female": 0.365546,
            "Sex_male": 0.634454,
            "Embarked_C": 0.182073,
            "Embarked_Q": 0.039216,
            "Embarked_S": 0.775910,
            "Pclass_Pclass": 5.704482,
            "Pclass_Age": 61.938151,
            "Pclass_SibSp": 1.198880,
            "Pclass_Parch": 0.983193,
            "Pclass_Fare": 53.052327,
            "Pclass_Sex_female": 0.754902,
            "Age_Age": 1092.761169,
            "Age_SibSp": 11.066415,
            "Age_Parch": 10.470476,
            "Age_Fare": 1104.142053,
            "Age_Sex_female": 10.204482,
            "SibSp_SibSp": 1.126050,
            "SibSp_Parch": 0.525210,
            "SibSp_Fare": 24.581262,
            "SibSp_Sex_female": 0.233894,
            "Parch_Parch": 0.913165,
            "Parch_Fare": 24.215465,
            "Parch_Sex_female": 0.259104,
            "Fare_Fare": 4000.200255,
            "Fare_Sex_female": 17.393698,
            "Sex_female_Sex_female": 0.365546
        }

        self.std = {
            "Pclass": 0.838250,
            "Age": 14.526497,
            "SibSp": 0.929783,
            "Parch": 0.853289,
            "Fare": 52.918930,
            "Sex_female": 0.481921,
            "Sex_male": 0.481921,
            "Embarked_C": 0.386175,
            "Embarked_Q": 0.194244,
            "Embarked_S": 0.417274,
            "Pclass_Pclass": 3.447593,
            "Pclass_Age": 34.379609,
            "Pclass_SibSp": 2.603741,
            "Pclass_Parch": 2.236945,
            "Pclass_Fare": 52.407209,
            "Pclass_Sex_female": 1.118572,
            "Age_Age": 991.079188,
            "Age_SibSp": 19.093099,
            "Age_Parch": 29.164503,
            "Age_Fare": 1949.356185,
            "Age_Sex_female": 15.924481,
            "SibSp_SibSp": 3.428831,
            "SibSp_Parch": 1.561298,
            "SibSp_Fare": 70.185369,
            "SibSp_Sex_female": 0.639885,
            "Parch_Parch": 3.008314,
            "Parch_Fare": 77.207321,
            "Parch_Sex_female": 0.729143,
            "Fare_Fare": 19105.110593,
            "Fare_Sex_female": 43.568303,
            "Sex_female_Sex_female": 0.481921
        }

        self.data = self._load_data()
        self.feature_size = len(self.data.columns) - 1

    def _load_data(self):
        df = pd.read_csv(self.file_path)
        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
        df = df.dropna(subset=["Age"])
        df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)
        base_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_female"]

        for i in range(len(base_features)):
            for j in range(i, len(base_features)):
                df[base_features[i] + "_" + base_features[j]] = ((df[base_features[i]] * df[base_features[j]]
                                                                  - self.mean[
                                                                      base_features[i] + "_" + base_features[j]])
                                                                 / self.std[base_features[i] + "_" + base_features[j]])
        for i in range(len(base_features)):
            df[base_features[i]] = (df[base_features[i]] - self.mean[base_features[i]]) / self.std[base_features[i]]
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.drop(columns=["Survived"]).iloc[idx].values
        label = self.data["Survived"].iloc[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
