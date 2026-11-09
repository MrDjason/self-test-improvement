# 冰湖游戏

## 1.介绍

链接：  

[冰湖游戏](https://gymnasium.org.cn/environments/toy_text/frozen_lake/ "跳转到冰湖游戏")

FrozenLake-v1 是 Gymnasium 中最经典的网格型（Grid-world）强化学习入门环境，整体为 4×4 冰面格子世界，智能体需要从左上角起点0出发，在不坠入冰洞的前提下抵达右下角15的目标终点，成功到达则获得奖励 1，坠入冰洞或中途失败则奖励为 0。  



与传统、理想化的格子寻路游戏不同，冰湖最大的设计特色，就是加入了**冰面打滑**（slippery） 随机机制：

- 在传统走格子游戏中，智能体的动作是完全确定的，只要规划出最短安全路径，就能稳定重复成功

- 而开启 slippery 后，选择一个方向后，只有 1/3 概率按指令正常移动，剩余 2/3 概率会随机**向左右两侧滑动偏移**

## 2.环境搭建

```python
# 导入库
import gymnasium as gym
from collections import defaultdict
import numpy as np

# 定义游戏
slippery = False
env = gym.make('FrozenLake-v1', 
               render_mode='human', # 设置以可视化运行
               is_slippery=slippery)

# 开始游戏
state, info = env.reset() # 初始化游戏
print(f'state:{state},info:{info}')

total_reward = 0

for _ in range(100):
    done = False
    while not done:
        action = env.action_space.sample() # 从动作空间随机采取行动
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated # 超时或结束则终止循环
    env.reset() # 每次游戏结束后初始化游戏

print(f'total_reward:{total_reward}')
```

这样，一个初步的冰湖游戏就搭建成功了。通过action = env.action_space.sample()从冰湖游戏所有动作中随机采取行动，大约有百分之一的概率在 slippery=False 情况下到达终点  



## 3.Slippery=False的冰湖游戏

### 3.1思路

### 3.1.1 确定性状态转移

冰湖游戏中state是一个0到15的数字，先将其转换成坐标，再与action进行计算

𝑠′ =𝑓⁡(𝑠,𝑎)

**计算坐标→移动→越界判断→返回新状态**



### 3.1.2 单步策略评估



### 3.1.3 迭代策略