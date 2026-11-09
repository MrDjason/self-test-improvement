import numpy as np
from collections import defaultdict
import res.gridworld_render as render_helper # 可视化 忽略

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

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)

# 单步策略评估
def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states(): # 遍历所有状态
        if state == env.goal_state: # 目的地的价值函数总是为0 智能代理抵达目的地 回合结束
            V[state] = 0
            continue

        action_probs = pi[state]
        new_V = 0

        for action, action_prob in action_probs.items():
        # 遍历状态s下所有可能的动作，按策略概率 π(a|s) 对「即时奖励 + 折扣后的下一状态价值」加权求和，得到状态s的新价值 V(s)
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state]) # 确定性状态迁移的价值函数更新式
        V[state] = new_V
    return V

# 迭代式策略评估
def policy_eval(pi, V, env, gamma, threshold=0.001):
    while True:
        old_V = V.copy() # 1.备份
        V = eval_onestep(pi, V, env, gamma) # 2.更新

        delta = 0 # 3.度量 算最大变化量
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold: # 4.收敛 V里最大的变化量小于0.001，就认为收敛了
            break
    return V

def argmax(d):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key

def greedy_policy(V, env, gamma):
    pi = {}

    for state in env.states(): # 遍历状态
        action_values = {}

        for action in env.actions(): # 遍历动作
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value
        max_action = argmax(action_values)
        action_probs = {0:0, 1:0, 2:0, 3:0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    return pi

def policy_iter(env, gamma, threshold=0.001, is_render=True):
    pi = defaultdict(lambda:{0:0.25, 1:0.25, 2:0.25, 3:0.25})
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold) # 评估
        new_pi =  greedy_policy(V, env, gamma) # 改进

        if is_render:
            env.render_v(V, pi)

        if new_pi == pi: # 更新检查
            break
        pi = new_pi

    return pi

def value_iter_onestep(V, env, gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values = []
        for action in env.actions(): # s' r v
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)

        V[state] = max(action_values)
    return V

def value_iter(V, env, gamma, threshold=0.001, is_render = True):
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V

env = GridWorld()
gamma = 0.9

V = defaultdict(lambda: 0)
V = value_iter(V, env, gamma)
pi = greedy_policy(V, env, gamma)
env.render_v(V, pi)