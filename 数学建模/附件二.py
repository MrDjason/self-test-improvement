import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler  # 改用 StandardScaler

import os

# ==============================
# 设置中文字体
# ==============================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================
# 1. 读取数据
# ==============================
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '核桃油VOCs清洗后数据.csv')
df = pd.read_csv(file_path)

time_cols = ['day_0', 'day_1', 'day_2', 'day_3', 'day_5', 'day_7', 'day_10']
time_days = [0, 1, 2, 3, 5, 7, 10]

# 转换为长格式
records = []
for _, row in df.iterrows():
    oil = row['oil_type']
    cat = row['category']
    comp = row['compound']
    for col, day in zip(time_cols, time_days):
        val = row[col]
        if pd.notna(val) and val != '':
            records.append([oil, cat, comp, day, float(val)])
df_long = pd.DataFrame(records, columns=['oil_type', 'category', 'compound', 'time', 'concentration'])

# ==============================
# 2. 按油品、类别、时间汇总求和
# ==============================
grouped = df_long.groupby(['oil_type', 'category', 'time'])['concentration'].sum().reset_index()

# ==============================
# 3. 标准化（Z-score）: 每个油品-类别组合独立进行
# ==============================
def standardize_group(group):
    """对每个油品-类别组合的时间序列进行 Z-score 标准化"""
    scaler = StandardScaler()
    standardized = scaler.fit_transform(group[['concentration']])
    group = group.copy()
    group['concentration_std'] = standardized.flatten()
    return group

grouped_std = grouped.groupby(['oil_type', 'category']).apply(standardize_group).reset_index(drop=True)

# ==============================
# 4. 对每种油分别绘图（使用标准化后的浓度）
# ==============================
oil_types = grouped_std['oil_type'].unique()
categories = grouped_std['category'].unique()

color_map = {
    '醇': 'red', '醛': 'blue', '酯': 'green',
    '酸': 'orange', '酮': 'purple', '其他': 'brown'
}
marker_map = {
    '醇': 'o', '醛': 's', '酯': '^',
    '酸': 'D', '酮': 'v', '其他': 'p'
}

for oil in oil_types:
    plt.figure(figsize=(10, 6))
    oil_data = grouped_std[grouped_std['oil_type'] == oil]
    
    for cat in categories:
        data = oil_data[oil_data['category'] == cat]
        if data.empty:
            continue
        data = data.sort_values('time')
        plt.scatter(data['time'], data['concentration_std'],
                    label=cat, color=color_map.get(cat, 'gray'),
                    marker=marker_map.get(cat, 'o'), s=60)
        plt.plot(data['time'], data['concentration_std'],
                 color=color_map.get(cat, 'gray'), linestyle='--', alpha=0.7)
    
    plt.title(f'Oil: {oil} - Standardized total VOCs concentration by category', fontsize=14)
    plt.xlabel('Time (day)', fontsize=12)
    plt.ylabel('Z-score (standard deviation units)', fontsize=12)
    # 标准化后纵轴自动调整范围，无需设置 ylim
    plt.legend(title='Category')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'VOCs_summary_standardized_{oil}.png', dpi=300)
    plt.show()

print("标准化图片已生成：VOCs_summary_standardized_香玲.png, 鲁光.png, 山核桃.png")