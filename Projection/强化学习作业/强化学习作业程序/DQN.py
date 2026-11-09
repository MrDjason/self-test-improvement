import gymnasium as gym
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt

import copy
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

# 经验回放器
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data]) # 沿新轴将一系列数组堆叠拼接起来
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done


# 神经网络QNet类
class QNet(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        # xW1 + b1，(N, 4) @ (4, 128) + (128,) → (N, 128)
        x = F.relu(self.l1(x))   # (N, 4)  → (N, 128)
    
        # xW2 + b2，(N, 128) @ (128, 128) + (128,) → (N, 128)
        x = F.relu(self.l2(x))   # (N, 128) → (N, 128)
    
        # xW3 + b3，(N, 128) @ (128, 2) + (2,) → (N, 2)
        x = self.l3(x)           # (N, 128) → (N, 2)
    
        # 每行对应一个样本：[Q(s, a0), Q(s, a1)]
        return x                 # (N, 2)

# DQN智能代理
class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet) # 设置qnet

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :] # state (4,) → (1, 4)
            qs = self.qnet(state) # (1, 2)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.get_batch() # 解包 batch_size × 4(或1) 列
        qs = self.qnet(state) # 传入 state (32, 4) 得到 qs (32, 2)
        # np.arange(x)生成[0,...,x-1]列表 action (32, 2)
        q = qs[np.arange(self.batch_size), action] # 花式索引,按坐标点成对抽取 (0, action[0]), (1, action[1]), ..., (31, action[31])

        next_qs = self.qnet_target(next_state) # 传入 next_state (32, 4) 得到 next_qs (32, 2)
        next_q = next_qs.max(axis=1) # (32,) 装着每一行最大的q值
        next_q.unchain()
        target = reward + (1 - done) * self.gamma * next_q

        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

episodes = 300
sync_interval= 20
render_mode = [None, 'human']
env = gym.make('CartPole-v1', render_mode = render_mode[0])
agent = DQNAgent()

reward_history= []

for episode in range(episodes):
    if episode % 10 == 0:
        print(f'{episode}')
    state, info = env.reset() # state = [小车位置, 小车速度, 木棒角度, 木棒角速度]
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)
print(f'{episodes}')
plt.plot(range(len(reward_history)), reward_history)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
env.close()
