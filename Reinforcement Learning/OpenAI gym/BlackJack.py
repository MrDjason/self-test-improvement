import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from collections import defaultdict

rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

env = gym.make('Blackjack-v1', natural=False, sab=False)
state, _ = env.reset()

# 改正策略
total_reward = 0
rewards = []

# Q 表
Q = defaultdict(lambda:np.zeros(env.action_space.n))
returns = defaultdict(list)


# 贪婪策略
def epsilon_greedy(state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])


num_episodes = 50000
gamma = 1.0

for episode in range(num_episodes):
    episodes_memory = []
    state, _ = env.reset()
    done = False
    while not done:
        # ========== 生成一局游戏 ==========

        action = epsilon_greedy(state)
        if state[0] < 12:
            action = 1
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        episodes_memory.append((state, action, reward))

        state = next_state
        done = terminated or truncated

    # ========== 回报计算 ==========
    G = 0
    visited = set()

    for state, action, reward in reversed(episodes_memory):
        G = gamma * G + reward
        if (state, action) not in visited:
            returns[(state, action)].append(G)
            Q[state][action] = np.mean(returns[state, action])
            visited.add((state, action))

    rewards.append(total_reward)
    env.reset()





print(f'总奖励点数：{total_reward}')

plt.plot(range(len(rewards)), rewards)
plt.xlabel('局数')
plt.ylabel('总奖励')
plt.title('改进策略')
plt.show()
env.close()