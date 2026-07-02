import gymnasium as gym
from collections import defaultdict
import numpy as np

slippery = False
env = gym.make('FrozenLake-v1', 
               # render_mode = 'human',
               is_slippery = slippery)

'''
def greedy_probs(Q, state, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)
    action_probs = {action: 0.0 for action in range(action_size)}
    action_probs[max_action] = 1
    return action_probs
'''

def greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)
    if max(qs) == min(qs):
        max_action = np.random.choice(action_size)
    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)} # action_probs = {0:ε/4,, 1:ε/4, 2:ε/4, 3:ε/4}
    action_probs[max_action] += (1 - epsilon)
    return action_probs

class RandomAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        self.alpha = 0.1
        self.epsilon = 0.1

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    '''
    def eval(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.Q[state] += (G - self.V[state]) / self.cnts[state]
    '''

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)
            self.cnts[key] += 1
            # self.Q[key] += (G - self.Q[key]) / self.cnts[key]

            self.Q[key] += (G - self.Q[key]) * self.alpha
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)

agent = RandomAgent()
total_reward = 0

for _ in range(1000):
    state, _ = env.reset()
    agent.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.add(state, action, reward)
        total_reward += reward
        done = terminated or truncated
        state = next_state        
    
    agent.update()


print(f'total_reward:{total_reward}')
env.close()