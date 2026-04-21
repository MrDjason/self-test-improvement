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