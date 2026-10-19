import gymnasium as gym
import numpy as np
from collections import defaultdict

slippery = False
env = gym.make('FrozenLake-v1',
               render_mode='human',
               is_slippery=slippery)

state, info = env.reset()
print(f'state:{state},info:{info}')

total_reward = 0

for _ in range(100):
    done = False
    while not done:
        action = env.action_space.sample() # 从动作空间随即采取行动
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated # 超时或结束则终止循环
    env.reset() # 每次游戏结束后初始化游戏

print(f'total_reward:{total_reward}')

