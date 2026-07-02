# 冰湖游戏

## 介绍

链接：  
[冰湖游戏](https://gymnasium.org.cn/environments/toy_text/frozen_lake/ "跳转到冰湖游戏")

FrozenLake-v1 是 Gymnasium 中最经典的网格型（Grid-world）强化学习入门环境，整体为 4×4 冰面格子世界，智能体需要从左上角起点0出发，在不坠入冰洞的前提下抵达右下角15的目标终点，成功到达则获得奖励 1，坠入冰洞或中途失败则奖励为 0。  
与传统、理想化的格子寻路游戏不同，冰湖最大的设计特色，就是加入了冰面打滑（slippery） 随机机制：
- 在传统走格子游戏中，智能体的动作是完全确定的，只要规划出最短安全路径，就能稳定重复成功
- 而开启 slippery 后，选择一个方向后，只有 1/3 概率按指令正常移动，剩余 2/3 概率会随机向左右两侧滑动偏移

## 参数

```python
n_actions = env.action_space.n # 4种行动
n_states = env.observation_space.n # 16个格子
goal_state = n_states - 1 # 终点索引 15

pi = defaultdict(lambda:{0:0.25, 1:0.25, 2:0.25, 3:0.25}) # 定义初始策略pi，pi为default类型dict，每个键值对分别对应state(0-15)和方向概率
V = defaultdict(lambda: 0) # 定义初始价值函数，V为default类型dict，每个值初始化为0
```

## 环境搭建

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

## Slippery=False的冰湖游戏

### 1.确定性状态转移

冰湖游戏中state是一个0到15的数字，先将其转换成坐标，再与action进行计算

$s' = f(s,a)$

**计算坐标→移动→越界判断→返回新状态**

```python
def next_state(state, action):
    row, col = state // 4, state % 4
    action_move_map = [(0,-1), (1,0), (0,1), (-1,0)] # 左下右上
    next_row, next_col = row + action_move_map[action][0], col + action_move_map[action][1]
    if not (0<=next_row<=3 and 0<=next_col<=3):
        next_row, next_col = row, col
    return next_row * 4 + next_col
```

### 2.单步策略评估

```python
# 单步策略评估
def eval_onestep(pi, V, gamma):
    for state in range(n_state):
        if state == goal_state or state in holes:
            V[state] = 0
            continue
        new_V = 0
        action_probs = pi[state]
        for action,action_prob in action_probs.items():
            next_s = next_state(state, action)
            r = 1 if next_s == goal_state else 0
            new_V += action_prob * (r + gamma * V[next_s])
        V[state] = new_V
    return V
```

### 3.迭代策略

```python
def policy_eval(pi, V, gamma, threshold=0.001):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
                break

    return V
```

### 4.最大索引
```python
# 最大索引
def argmax(d):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key
```

### 5.贪婪策略
```python

# 贪婪策略
def greedy_policy(V, gamma):
    pi = {}
    for state in range(n_state):
        action_values = {}
        value = 0
        for action in range(n_action):
            next_s = next_state(state, action)
            r = 1 if next_s == goal_state else 0
            value = r + gamma * V[next_s]
            action_values[action] = value

        max_action = argmax(action_values)
        action_probs = {0:0, 1:0, 2:0, 3:0}
        action_probs[max_action] = 1
        pi[state] = action_probs
    return pi
```

### 6.完整程序
```python
import gymnasium as gym
from collections import defaultdict
import numpy as np

# 定义游戏
slippery = False
env = gym.make('FrozenLake-v1', 
               #render_mode='human', # 设置以可视化运行
               is_slippery=slippery)

n_action = env.action_space.n
n_state = env.observation_space.n
goal_state = n_state - 1

holes = [5, 7, 11, 12]

# 移动
def next_state(state, action):
    row, col = state // 4, state % 4
    action_move_map = [(0,-1), (1,0), (0,1), (-1,0)] # 左下右上
    next_row, next_col = row + action_move_map[action][0], col + action_move_map[action][1]
    if not (0<=next_row<=3 and 0<=next_col<=3):
        next_row, next_col = row, col
    return next_row * 4 + next_col

# 单步策略评估
def eval_onestep(pi, V, gamma):
    for state in range(n_state):
        if state == goal_state or state in holes:
            V[state] = 0
            continue
        new_V = 0
        action_probs = pi[state]
        for action,action_prob in action_probs.items():
            next_s = next_state(state, action)
            r = 1 if next_s == goal_state else 0
            new_V += action_prob * (r + gamma * V[next_s])
        V[state] = new_V
    return V

# 迭代式策略评估
def policy_eval(pi, V, gamma, threshold=0.001):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
                break

    return V

# 最大索引
def argmax(d):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key

# 贪婪策略
def greedy_policy(V, gamma):
    pi = {}
    for state in range(n_state):
        action_values = {}
        value = 0
        for action in range(n_action):
            next_s = next_state(state, action)
            r = 1 if next_s == goal_state else 0
            value = r + gamma * V[next_s]
            action_values[action] = value

        max_action = argmax(action_values)
        action_probs = {0:0, 1:0, 2:0, 3:0}
        action_probs[max_action] = 1
        pi[state] = action_probs
    return pi

# 开始游戏
gamma = 0.95
total_reward = 0
pi = defaultdict(lambda:{0:0.25, 1:0.25, 2:0.25, 3:0.25})
V = defaultdict(lambda: 0)

state, info = env.reset() # 初始化游戏
print(f'state:{state},info:{info}')
V = policy_eval(pi, V, gamma) # 迭代策略
pi = greedy_policy(V,gamma) # 
for _ in range(100):
    done = False
    while not done:
        action = argmax(pi[state])
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated # 超时或结束则终止循环
    env.reset() # 每次游戏结束后初始化游戏

print(f'total_reward:{total_reward}')
```

## Slicppery=True的策略迭代冰湖游戏

策略迭代法交替重复评估和改进

```python
import gymnasium as gym
from collections import defaultdict
import numpy as np

# ========== 定义游戏 ==========
slippery=True
env = gym.make('FrozenLake-v1', 
               #render_mode='human', 
               is_slippery=slippery)
# ========== 全局变量 ==========
total_reward = 0
state, info = env.reset()
print(f'state:{state},info:{info}') # 状态和状态转移概率

n_actions = env.action_space.n # 4
n_states = env.observation_space.n # 16
goal_state = n_states - 1 # 终点 15

holes = [5, 7, 11, 12]

gamma = 0.95
# ========== 移动 ==========
def next_state(state, action):
    # 位置
    row, col = state // 4, state % 4
    action_move_map = [(0,-1), (1,0), (0,1), (-1,0)] # 进行的是行列移动非坐标系
    move = action_move_map[action]
    next_row, next_col = row + move[0], col + move[1]
    
    if not (0 <= next_row <= 3 and 0 <= next_col <= 3):
        next_row, next_col = row, col
    return (next_row * 4 + next_col)

# ========== Slippery偏移 ==========
def get_transitions(state, action):
    if state in holes or state == goal_state:
        return [(state, 1.0)]

    slip_actions = [action, (action-1)%4, (action+1)%4] # 左下右上 可能滑动方向
    transitions = []

    for action in slip_actions:
        next_s = next_state(state, action)
        transitions.append((next_s, 1/3))

    return transitions

# ========== 单步策略评估 ==========
def eval_onestep(pi, V, gamma=0.9):

    for state in range(n_states):
        if state == goal_state or state in holes:
            V[state] = 0
            continue
        
        new_V = 0 # 
        action_probs = pi[state] # 

        for action, action_prob in action_probs.items():
            if slippery:
                transitions = get_transitions(state, action)
                for next_s, trans_prob in transitions:
                    r = 1 if next_s == goal_state else 0
                    new_V += action_prob * trans_prob * (r + gamma * V[next_s])
            else:
                next_s = next_state(state, action)
                r = 1 if next_s == goal_state else 0
                new_V += action_prob * (r + gamma * V[next_s])
        V[state] = new_V
    return V

# ========== 迭代式策略评估 ==========
def policy_eval(pi, V, gamma, threshold=0.001):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
        
        if delta < threshold:
            break

    return V

# ========== 最大索引 ==========
def argmax(d):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key

# ========== 贪婪策略 ==========
def greedy_policy(V, gamma):
    pi = {}

    for state in range(n_states):
        action_values = {}

        for action in range(n_actions):
            if slippery:
                value = 0
                transitions = get_transitions(state, action)
                for next_s, trans_prob in transitions:
                    r = 1 if next_s == goal_state else 0
                    value += trans_prob * (r + gamma * V[next_s])
                    action_values[action] = value
            else:
                next_s = next_state(state, action)
                r = 1 if next_s == goal_state else 0
                value = r + gamma * V[next_s]
                action_values[action] = value
        max_action = argmax(action_values)
        action_probs = {0:0, 1:0, 2:0, 3:0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi

# ========== 策略迭代 ==========
def policy_iter(gamma, threshold=0.001):
    pi = defaultdict(lambda:{0:0.25, 1:0.25, 2:0.25, 3:0.25})
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, gamma, threshold)
        new_pi = greedy_policy(V, gamma)
        new_pi = defaultdict(lambda:{0:0.25, 1:0.25, 2:0.25, 3:0.25}, new_pi) # 将dict类型的new_pi转为defaultdict类型
        if new_pi == pi:
            break
        pi = new_pi
    return pi

# ========== 运行游戏 ==========
pi = policy_iter(gamma)

for _ in range(100):
    state, _ = env.reset()
    done = False
    while not done:
        action = argmax(pi[state])
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    env.reset()
print(f'total_reward:{total_reward}')
env.close()
```
## Slicppery=True的价值迭代冰湖游戏

---

```python
import gymnasium as gym
from collections import defaultdict
import numpy as np

# 定义游戏
slippery = True
env = gym.make('FrozenLake-v1', 
               #render_mode='human', # 设置以可视化运行
               is_slippery=slippery)

n_action = env.action_space.n
n_state = env.observation_space.n
goal_state = n_state - 1

holes = [5, 7, 11, 12]

def next_state(state, action):
    row, col = state // 4, state % 4
    action_move_map = [(0,-1), (1,0), (0,1), (-1,0)]
    next_row, next_col = row + action_move_map[action][0], col + action_move_map[action][1]
    if not (0<=next_row<=3 and 0<=next_col<=3):
        next_row, next_col = row, col

    return next_row * 4 + next_col

# 状态转移
def get_transitions(state, action):
    if state == goal_state or state in holes:
        return [(state,1.0)]
    sli_actions = [action, (action-1)%4, (action+1)%4]
    transitions = []
    for action in sli_actions:
        next_s = next_state(state, action)
        transitions.append((next_s, 1/3))
    return transitions

def argmax(d):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key

def greedy_policy(V, gamma):
    pi = {}
    for state in range(n_state):
        action_values = {}
        for action in range(n_action):
            if slippery:
                transitions = get_transitions(state, action)
                value = 0
                for next_s, trans_prob in transitions:
                    r = 1 if next_s == goal_state else 0
                    value += trans_prob * (r + gamma * V[next_s])
                action_values[action] = value
            else:
                next_s = next_state(state, action)
                r = 1 if next_s == goal_state else 0
                value = r + gamma * V[next_s]
                action_values[action] = value
        max_action = argmax(action_values)
        action_probs = {0:0, 1:0, 2:0, 3:0}
        action_probs[max_action] = 1
        pi[state] = action_probs
    return pi

def value_iter_onestep(V, gamma):
    for state in range(n_state):
        if state == goal_state or state in holes:
            V[state] = 0
            continue

        action_values = []
        for action in range(n_action):
            if slippery:
                value = 0
                transitions = get_transitions(state, action)
                for next_s, trans_prob in transitions:
                    r = 1 if next_s == goal_state else 0
                    value += trans_prob * (r + gamma * V[next_s])
                action_values.append(value)
            else:
                next_s = next_state(state, action)
                r = 1 if next_s == goal_state else 0
                value = r + gamma * V[next_s]
                action_values.append(value)

        V[state] = max(action_values)
    return V

def value_iter(V, gamma, threshold=0.001):
    while True:
        old_V = V.copy()
        V = value_iter_onestep(V, gamma)
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
        if delta < threshold:
            break
    return V

# 开始游戏
gamma = 0.90
total_reward = 0
pi = defaultdict(lambda:{0:0.25, 1:0.25, 2:0.25, 3:0.25})
V = defaultdict(lambda: 0)

state, info = env.reset() # 初始化游戏
print(f'state:{state},info:{info}')

V = value_iter(V, gamma)
pi = greedy_policy(V, gamma)

for _ in range(100):
    done = False
    while not done:
        action = argmax(pi[state])
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated # 超时或结束则终止循环
    state, _ = env.reset() # 每次游戏结束后初始化游戏

print(f'total_reward:{total_reward}')
```

我把**整段代码拆成9个核心模块**，每个模块都用 **大白话描述 + 核心数学公式** 讲透，完全对应你写的代码逻辑，看完就能把「代码→文字→公式」彻底打通！

全程基于**冰湖游戏（FrozenLake）**：
16个状态（0起点，15终点，5/7/11/12陷阱）、4个动作（左0/下1/右2/上3）、冰面可打滑（随机环境）

---

# 模块1：库导入 + 游戏环境初始化
## 语言描述
导入强化学习环境库、字典工具、数值库；创建冰湖游戏，开启/关闭冰面打滑效果。
## 代码对应
```python
import gymnasium as gym
from collections import defaultdict
import numpy as np

slippery = True
env = gym.make('FrozenLake-v1', is_slippery=slippery)
```

# 模块2：游戏核心常量定义
## 语言描述
定义游戏固定参数：动作数量、状态数量、终点状态、陷阱位置，是所有算法的基础参数。
## 代码对应
```python
n_action = env.action_space.n  # 动作数=4
n_state = env.observation_space.n  # 状态数=16
goal_state = n_state - 1  # 终点=15
holes = [5, 7, 11, 12]  # 陷阱
```
# 模块3：理想状态转移函数 `next_state(state, action)`
## 语言描述
**不考虑打滑**：输入当前状态+动作，计算「理论上应该到达的下一个状态」；出界则原地不动。
## 核心公式
**确定性状态转移**
$$ s' = f(s, a) $$
- $s$：当前状态
- $a$：当前动作
- $s'$：确定的下一个状态
- $f$：坐标移动规则（4x4网格）
## 代码对应
计算坐标→移动→越界判断→返回新状态，是所有转移的基础。

# 模块4：概率状态转移函数 `get_transitions(state, action)`
## 语言描述
**考虑打滑（随机环境）**：执行动作后，有3种等概率结果（目标动作+左滑+右滑）；终点/陷阱状态固定不动。
## 核心公式
**随机状态转移概率**
$$ P(s'|s,a) = \begin{cases}
1 & s是终点/陷阱 \\
1/3 & 打滑的3种可能状态
\end{cases} $$
- $P(s'|s,a)$：在状态$s$执行动作$a$，转移到$s'$的概率
## 代码对应
返回 `[(下一个状态, 概率), ...]`，是值迭代处理随机环境的核心。

# 模块5：最优动作选择函数 `argmax(d)`
## 语言描述
输入「动作:价值」字典，**选出价值最大的动作**（贪心选择）。
## 核心公式
$$ a^* = \arg\max_a Q(s,a) $$
- $a^*$：最优动作
- $Q(s,a)$：当前状态$s$下动作$a$的价值
## 代码对应
遍历字典，找到最大值对应的键（动作）。

---

# 模块6：贪心策略生成函数 `greedy_policy(V, gamma)`
## 语言描述
根据**最优状态价值$V$**，计算每个状态的**最优动作**，生成贪心策略：每个状态只选价值最高的动作（概率=1）。
分两种场景：
1. **打滑（随机）**：按概率加权计算动作价值
2. **不打滑（确定）**：直接计算动作价值

## 核心公式
1. **确定性动作价值**
$$ Q(s,a) = r + \gamma \cdot V(s') $$
2. **随机性动作价值（数学期望）**
$$ Q(s,a) = \sum_{s'} P(s'|s,a) \cdot \big[ r + \gamma \cdot V(s') \big] $$
3. **贪心策略**
$$ \pi(a|s) = \begin{cases}
1 & a = \arg\max_a Q(s,a) \\
0 & 其他动作
\end{cases} $$
- $r$：奖励（终点=1，其他=0）
- $\gamma$：折扣因子
- $\pi(a|s)$：状态$s$下选择动作$a$的概率
## 代码对应
遍历所有状态→计算每个动作价值→选最优→生成策略字典。

---

# 模块7：单步值迭代函数 `value_iter_onestep(V, gamma)`
## 语言描述
**值迭代核心更新步骤**：对所有状态更新一次价值；
终点/陷阱价值固定为0（`continue`跳过无效计算）；
状态价值 = 该状态下**所有动作价值的最大值**。

## 核心公式
**值迭代状态更新公式（贝尔曼最优方程）**
$$ V(s) = \max_{a} Q(s,a) $$
结合动作价值：
1. 确定环境：$V(s) = \max_a \left[ r + \gamma V(s') \right]$
2. 随机环境：$V(s) = \max_a \left[ \sum_{s'} P(s'|s,a) \cdot \left(r+\gamma V(s')\right) \right]$
## 代码对应
遍历所有状态→计算4个动作价值→取最大值更新$V(s)$。

---

# 模块8：值迭代主函数 `value_iter(V, gamma)`
## 语言描述
**反复执行单步值迭代**，直到状态价值变化极小（收敛）；
用`delta`记录最大价值变化，小于阈值则停止迭代，得到**最优状态价值$V^*$**。
## 核心公式
**收敛条件**
$$ \max_s |V_{new}(s) - V_{old}(s)| < \theta $$
- $\theta=0.001$：收敛阈值
- $V_{new}$：更新后的价值
- $V_{old}$：更新前的价值
## 代码对应
循环调用单步更新→计算价值差→满足阈值则退出。

---

# 模块9：主程序 + 策略测试
## 语言描述
1. 初始化价值$V$（全0）、折扣因子$\gamma=0.9$
2. 运行值迭代，得到最优价值$V^*$
3. 生成最优贪心策略$\pi^*$
4. 用最优策略玩100局游戏，统计总得分（成功到达终点的次数）
## 核心公式
最优状态价值：$V^* = value\_iter(V, \gamma)$  
最优策略：$\pi^* = greedy\_policy(V^*, \gamma)$
## 代码对应
初始化→迭代→测策略→输出得分。
