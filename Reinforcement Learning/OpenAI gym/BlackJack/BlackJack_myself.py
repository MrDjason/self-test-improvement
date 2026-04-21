import gymnasium as gym
import numpy as np
from collections import defaultdict


env = gym.make('Blackjack-v1')

state, _ = env.reset()

total_reward = 0
episodes = 10000
gamma = 1.0

Q = defaultdict(lambda:np.zeros(env.action_space.n)) # 
returns = defaultdict(list) # 

def epsilon_greedy(state, epsilon=0.1):
    if (np.random.rand() < epsilon):
        return env.action_space.sample()
    return np.argmax(Q[state])

for episode in range(episodes):
    episode_memory = []
    state, _ = env.reset()
    done = False
    while not done:
        # ========== 生成一局游戏 ==========
        action = epsilon_greedy(state)
        if state[0] < 12:
            action = 1
        next_state, reward, terminated, truncated, info=env.step(action)
        total_reward += reward
        episode_memory.append((state, action, reward))
        state = next_state
        done = terminated or truncated


    # ========== 奖励 ========== 重要
    G = 0
    visited = set()

    for state, action, reward in reversed(episode_memory):
        G = gamma * G + reward
    
        if (state, action) not in visited:
            returns[(state, action)].append(G) # 给词典中键(state, action)添加值为G
            Q[state][action] = np.mean(returns[state, action]) # 给两个行动（停牌、要牌）算平均值

            visited.add((state, action))

    env.reset()
print(f'{total_reward}')

policy = {}
for state in Q:
    policy[state] = np.argmax(Q[state]) # 取得Q中状态价值最大的索引

# ========== 测试 ==========
total_reward = 0
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        action = policy[state]
        if state[0] < 12:
            action = 1
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    env.reset()

print(f'{total_reward}')
env.close()
