import gymnasium as gym
from collections import defaultdict
import numpy as np

slippery = False
total_reward = 0
episodes = 1000

env = gym.make('FrozenLake-v1',
               #render_mode = 'human',
               is_slippery=slippery)

class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4
        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda:0)

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def eval(self, state, reward, next_state, done):
        next_V = 0 if done else self.V[next_state]
        target = reward + self.gamma * next_V
        self.V[state] += (target - self.V[state]) * self.alpha

agent = TdAgent()
for _ in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        agent.eval(state, reward, next_state, done)
        state = next_state
    total_reward += reward
env.close()
print(f'total_reward:{total_reward}')