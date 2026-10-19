import numpy as np
from collections import defaultdict
import res.gridworld_render as render_helper

# GridWorld环境类
class GridWorld:
    def __init__(self):
        # 动作空间 动作空间定义 地图 各种地点定义
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = {
            '0' : 'UP',
            '1' : 'DOWN',
            '2' : 'LEFT',
            '3' : 'RIGHT'
        }

        self.reward_map = np.array(
            [[0, 0, 0, 1.0],
            [0, None, 0, -1.0],
            [0, 0, 0, 0]])

        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield(h, w)

    def next_state(self, state, action):
        # 1.移动目的地的计算
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        # 2.条件判断
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state
        return next_state

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_q(q, print_value)
# 智能代理类的实现
class RandomAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda: random_actions) # 初始策略为随机移动
        self.V = defaultdict(lambda: 0)               # 价值函数
        self.cnts = defaultdict(lambda: 0)            # 增量式求平均值处理
        self.memory = []                              # 实际行动中获得的经验 状态,行动,奖励这三数据列

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)     # 根据probs列表中的概率抽取actions列表中的行动

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)                      # 将(state, action, reward)元组汇总到列表中

    def reset(self):
        self.memory.clear()

    def eval(self):                                   # 执行蒙特卡洛方法
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]

# 蒙特卡洛方法进行策略控制 的 智能代理
def greedy_probs(Q, state, epsilon = 0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]          # 从Q字典解码对应 (state, action) 这个元组作为键所对应的值
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)} # 此时action_probs = {0:0.0, 1:0.0, 2:0.0, 3:0.0}
    action_probs[max_action] += (1 - epsilon)
    return action_probs

class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.1
        self.action_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda:0)                 # 唯一区别:使用Q而非V
        self.cnts = defaultdict(lambda:0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        data = (state, action, reward)
        self. memory.append(data)

    def reset(self):
        self.memory.clear()

    # 更新
    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward                         # 计算 G
            key = (state, action)
            self.cnts[key] += 1                                 # 记录这个状态动作对被访问的次数
            # self.Q[key] += (G - self.Q[key]) / self.cnts[key] # 状态s行动为a的行动价值函数Q增量计算
            self.Q[key] += (G - self.Q[key]) * self.alpha
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)        # 利用贪婪策略更新
'''
# 运行蒙特卡洛方法
env = GridWorld()
agent = RandomAgent()

episodes = 1000
for episode in range(episodes):
    # 重置位置和智能代理的记忆
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:
            agent.eval()
            break

        state = next_state
env.render_v(agent.V)
'''

env = GridWorld()
agent = McAgent()

episodes = 10000
for episode in range(episodes):
    if episode % 1000 == 0:
        print(f"episode {episode}")
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:
            agent.update()
            break

        state = next_state
env.render_q(agent.Q)