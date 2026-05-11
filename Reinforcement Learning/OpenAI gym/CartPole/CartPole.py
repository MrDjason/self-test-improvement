import gymnasium as gym
from collections import deque
import random
import numpy as np



env = gym.make('CartPole-v1', 
               render_mode='human',
               )

state, _ = env.reset() # state = [位置, 速度, 角度, 角速度]
episodes = 10

for episode in range(episodes):
    env.reset()
    done = False
    while not done:
        action = 0
        next_state, reward, truncated, terminated, _ = env.step(action)

        done = truncated or terminated
env.close()

