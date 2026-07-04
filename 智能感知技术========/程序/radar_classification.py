# -*- coding: utf-8 -*-
"""
毫米波雷达实测数据分类 — 三类目标识别
使用 SVM、决策树、随机森林对金杯(0)、厢式货车(1)、无人机(2)进行分类
"""

import numpy as np
import scipy.io as sio
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ======================== 1. 数据加载 ========================
BASE_DIR = r'c:\Users\13218\Documents\GitHub\self-test-improvement\智能感知技术========\数据'

# 三类目标：金杯=0, 厢式货车=1, 无人机=2
CATEGORIES = {
    '金杯':     {'label': 0, 'subdirs': ['0611']},
    '厢式货车': {'label': 1, 'subdirs': ['0610']},
    '无人机':   {'label': 2, 'subdirs': ['0610', '0612']},
}

def load_all_data(base_dir, categories):
    """
    加载所有 .mat 文件，提取特征和标签
    每个轨迹点作为一条样本，特征取 Col1~Col7
    """
    X_list, y_list = [], []
    file_count = 0

    for cls_name, info in categories.items():
        label = info['label']
        for subdir in info['subdirs']:
            dir_path = os.path.join(base_dir, cls_name, subdir)
            if not os.path.exists(dir_path):
                continue
            for fname in os.listdir(dir_path):
                if not fname.endswith('.mat'):
                    continue
                file_path = os.path.join(dir_path, fname)
                try:
                    data = sio.loadmat(file_path)
                    md = data['MatData']  # shape (N_points, 9)
                    n_points = md.shape[0]

                    # 提取 Col1~Col7 作为特征
                    features = np.zeros((n_points, 7))
                    for i in range(n_points):
                        for j in range(7):
                            features[i, j] = float(md[i, j + 1].item())

                    X_list.append(features)
                    y_list.append(np.full(n_points, label))
                    file_count += 1

                except Exception as e:
                    print(f"  警告: 读取 {fname} 失败: {e}")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    print(f"成功加载 {file_count} 个文件, 总样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    return X, y

def load_file_level_data(base_dir, categories):
    """
    以文件（航迹）为单位：每个文件的所有轨迹点特征取均值/标准差作为特征向量
    """
    X_list, y_list = [], []
    file_names = []
    file_count = 0

    for cls_name, info in categories.items():
        label = info['label']
        for subdir in info['subdirs']:
            dir_path = os.path.join(base_dir, cls_name, subdir)
            if not os.path.exists(dir_path):
                continue
            for fname in os.listdir(dir_path):
                if not fname.endswith('.mat'):
                    continue
                file_path = os.path.join(dir_path, fname)
                try:
                    data = sio.loadmat(file_path)
                    md = data['MatData']
                    n_points = md.shape[0]

                    features = np.zeros((n_points, 7))
                    for i in range(n_points):
                        for j in range(7):
                            features[i, j] = float(md[i, j + 1].item())

                    # 对每条航迹提取统计特征
                    feat_vec = np.concatenate([
                        np.mean(features, axis=0),
                        np.std(features, axis=0),
                        np.max(features, axis=0),
                        np.min(features, axis=0),
                    ])  # 28维特征

                    X_list.append(feat_vec)
                    y_list.append(label)
                    file_names.append(fname)
                    file_count += 1

                except Exception as e:
                    print(f"  警告: 读取 {fname} 失败: {e}")

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"文件级特征: {file_count} 个文件, 特征维度: {X.shape[1]}")
    return X, y, file_names

# ======================== 2. 加载数据 ========================
print("=" * 60)
print("加载轨迹点级数据...")
X_point, y_point = load_all_data(BASE_DIR, CATEGORIES)

print("\n加载文件级数据（航迹统计特征）...")
X_file, y_file, file_names = load_file_level_data(BASE_DIR, CATEGORIES)

# 打印数据分布
print(f"\n数据分布:")
for cls_name, info in CATEGORIES.items():
    n_point = np.sum(y_point == info['label'])
    n_file = np.sum(y_file == info['label'])
    print(f"  {cls_name} (类别{info['label']}): {n_point} 个轨迹点, {n_file} 条航迹")

# ======================== 3. 模型训练与评估 ========================
models = {
    'SVM (RBF核)': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    'SVM (线性核)': SVC(kernel='linear', C=1.0, random_state=42),
    '决策树': DecisionTreeClassifier(max_depth=5, random_state=42),
    '随机森林': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
}

print("\n" + "=" * 60)
print("====== 方案A: 轨迹点级分类（每条轨迹点独立分类）======")
print("=" * 60)

# 划分训练/测试集（7:3）
X_tr, X_te, y_tr, y_te = train_test_split(
    X_point, y_point, test_size=0.3, random_state=42, stratify=y_point
)
print(f"训练集: {X_tr.shape[0]} 样本, 测试集: {X_te.shape[0]} 样本")

# 标准化
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    model.fit(X_tr_scaled, y_tr)
    y_pred = model.predict(X_te_scaled)
    acc = accuracy_score(y_te, y_pred)

    # 5折交叉验证
    cv_scores = cross_val_score(model, X_tr_scaled, y_tr, cv=5, scoring='accuracy')

    print(f"\n--- {name} ---")
    print(f"  测试准确率: {acc:.4f}")
    print(f"  5折CV准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_te, y_pred)
    im = axes[idx].imshow(cm, cmap='Blues', interpolation='nearest')
    axes[idx].set_title(f'{name}\n准确率={acc:.3f}')
    axes[idx].set_xlabel('预测类别')
    axes[idx].set_ylabel('真实类别')
    cls_names = list(CATEGORIES.keys())
    axes[idx].set_xticks(range(len(cls_names)))
    axes[idx].set_xticklabels(cls_names)
    axes[idx].set_yticks(range(len(cls_names)))
    axes[idx].set_yticklabels(cls_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[idx].text(j, i, str(cm[i, j]), ha='center', va='center',
                          color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.colorbar(im, ax=axes[idx])

plt.suptitle('轨迹点级分类 — 混淆矩阵对比', fontsize=14)
plt.tight_layout()
plt.savefig('fig_classification_point_level.png', dpi=150)
plt.show()

# ======================== 4. 文件级分类 ========================
print("\n" + "=" * 60)
print("====== 方案B: 文件级分类（每条航迹为一个样本）======")
print("=" * 60)

Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(
    X_file, y_file, test_size=0.3, random_state=42, stratify=y_file
)
print(f"训练集: {Xf_tr.shape[0]} 条航迹, 测试集: {Xf_te.shape[0]} 条航迹")

scaler_f = StandardScaler()
Xf_tr_scaled = scaler_f.fit_transform(Xf_tr)
Xf_te_scaled = scaler_f.transform(Xf_te)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

best_model = None
best_acc = 0

for idx, (name, model) in enumerate(models.items()):
    model.fit(Xf_tr_scaled, yf_tr)
    yf_pred = model.predict(Xf_te_scaled)
    acc = accuracy_score(yf_te, yf_pred)

    if acc > best_acc:
        best_acc = acc
        best_model = (name, model)

    cv_scores = cross_val_score(model, Xf_tr_scaled, yf_tr, cv=5, scoring='accuracy')

    print(f"\n--- {name} ---")
    print(f"  测试准确率: {acc:.4f}")
    print(f"  5折CV准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(classification_report(yf_te, yf_pred, target_names=list(CATEGORIES.keys())))

    cm = confusion_matrix(yf_te, yf_pred)
    im = axes[idx].imshow(cm, cmap='Blues', interpolation='nearest')
    axes[idx].set_title(f'{name}\n准确率={acc:.3f}')
    axes[idx].set_xlabel('预测类别')
    axes[idx].set_ylabel('真实类别')
    cls_names = list(CATEGORIES.keys())
    axes[idx].set_xticks(range(len(cls_names)))
    axes[idx].set_xticklabels(cls_names)
    axes[idx].set_yticks(range(len(cls_names)))
    axes[idx].set_yticklabels(cls_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[idx].text(j, i, str(cm[i, j]), ha='center', va='center',
                          color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.colorbar(im, ax=axes[idx])

plt.suptitle('文件级（航迹）分类 — 混淆矩阵对比', fontsize=14)
plt.tight_layout()
plt.savefig('fig_classification_file_level.png', dpi=150)
plt.show()

# ======================== 5. 汇总 ========================
print("\n" + "=" * 60)
print("====== 分类结果汇总 =======")
print(f"最佳模型: {best_model[0]}, 准确率: {best_acc:.4f}")
print("\n结论:")
print("  1. 文件级（航迹统计特征）分类效果通常优于轨迹点级分类")
print("  2. 随机森林和SVM对三类目标的区分能力较强")
print("  3. 金杯(小型车)和厢式货车(大型车)的雷达特征存在一定相似性")
print("=" * 60)
