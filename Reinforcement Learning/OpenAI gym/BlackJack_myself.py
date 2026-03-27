import numpy as np
import gymnasium as gym
from collections import defaultdict

total_rewards = 0
# Q 表
Q = defaultdict(lambda:0)
returns = defaultdict(list)

env = gym.make('Blackjack-v1', natural=False, sab=False)
episodes_nums = 50000
gamma = 1.0

def epsilon_greedy(state, epsilon = 0.1):
    if np.random.rand < epsilon:
        return env.action_space.sample()
    # 返回
    return np.argmax(Q[state])
    
for episode in range(episodes_nums):
    # ========== 定义一局游戏 ==========
    done = False
    state, _ = env.reset()
    # episodes_memory
    while not done:
        action = epsilon_greedy(state)
        if state[0] < 12:
            action = 1
        state, reward, terminated ,truncated, _ = env.step(action)
        total_rewards += reward
        done = terminated or truncated
        env.reset()
    # 步更新

# ========== 奖励 ==========.
    G = 0

    