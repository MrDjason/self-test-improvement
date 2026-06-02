# k近邻算法

## 算法思路

- 1.已有带标签的训练数据（分布在特征空间中）
- 2.输入待分类向量，计算其与所有训练点的欧氏距离
- 3.选取距离最近的 k 个点（近邻）按这k个点的标签进行投票
- 4.得票最多的即为预测类别

## 举例

|电影名称|打斗镜头|接吻镜头|电影类型|
|:---|:---:|:---:|:---:|
|**California Man**|3|104|爱情片|
|**He's Not Really into Dudes**|2|100|爱情片|
|**Beautiful Woman**|1|81|爱情片|
|**Kevin Longblade**|101|10|动作片|
|**RoboS layer 3000**|99|5|动作片|
|**Amped II**|98|2|动作片|
|**?**|18|90|未知|

即使不知道这部电影名称和类型，通过打斗镜头和接吻镜头也可以知道这部影片的类型

![电影分类](../res/电影分类.jpg "电影分类")

- **1.构造数据集**

由于我们只想分类`？`电影，可以忽略电影名称。只专注于`打斗镜头`、`接吻镜头`、`电影类型`。其中打斗镜头、接吻镜头作为其特征值，电影类型作为其标签，其中爱情片标签为`Love`，动作片标签为`Act`

```python
import numpy as np
dataset = np.array([[3, 104],
                    [2, 100],
                    [1, 81],
                    [101, 10],
                    [99, 5],
                    [98, 2]])
labels = ['Love', 'Love', 'Love', 'Act', 'Act', 'Act']

k = 3
x = np.array([18, 90])
```

- **2.输入待分类向量，计算其与所有训练点的欧氏距离**

设：
- 待分类样本：$\mathbf{x} = (x_1, x_2, \dots, x_n)$  
- 训练集中第 $i$ 个样本：$\mathbf{x}^{(i)} = (x_1^{(i)}, x_2^{(i)}, \dots, x_n^{(i)})$

则 $\mathbf{x}$ 与 $\mathbf{x}^{(i)}$ 之间的**欧氏距离**为：

$$
d_i = \sqrt{ \sum_{j=1}^{n} \left( x_j - x_j^{(i)} \right)^2 }
$$


x为待分类向量


$x_j - x_j^{(i)}$
```python
diff = x - dataset # 差值矩阵
```
$(x_j - x_j^{(i)})^2$
```python
sq_diff = diff ** 2 # 平方
```

$\sum_{j=1}^{n} \left( x_j - x_j^{(i)} \right)^2$

```python
sum_sq = sq_diff.sum(axis=1)
```

$\sqrt{ \sum_{j=1}^{n} \left( x_j - x_j^{(i)} \right)^2 }$

```python
distances = np.sqrt(sum_sq) # 开方得到欧氏距离
```

- **3.选取距离最近的k个点（近邻）按这k个点的标签进行投票**

```python
class_count = {}
sorted_indices = distances.argsort() # 返回距离数组排序后的索引
for i in range(k): # k 即为选取点的个数
    vote_label = labels[sorted_indices[i]] # 获取标签
    class_count[vote_label] = class_count.get(vote_label, 0) + 1 # 
```

- **4.得票最多的即为预测类别**

```python
import operator
sorted_class_count = sorted(class_count.items(), key = operator.itemgetter(1), reverse = True)
print(f'得票最多的结果：{sorted_class_count[0][0]}')
```