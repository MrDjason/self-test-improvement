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

# ========== 随机策略 价值函数 ==========
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