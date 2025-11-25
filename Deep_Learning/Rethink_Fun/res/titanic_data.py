# ==================== 导入模块 ====================
import pandas as pd
import os
# ==================== 处理模块 ====================
pd.set_option('display.max_columns', None) # 打印时显示所有列

# 从CSV文件读取数据（确保有正确路径）
print('当前工作目录:', os.getcwd())

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.csv')
df = pd.read_csv(data_path)



# 去除不需要列
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# 去除Age缺失样本
df = df.dropna(subset=['Age'])

# 对sex和embarked做独热编码
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], dtype=int)

print(df.head(10))
