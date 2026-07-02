import gymnasium as gym
from collections import defaultdict, deque
import numpy as np
slippery = False
total_reward = 0
episodes = 1000

env = gym.make('FrozenLake-v1',
               #render_mode = 'human',
               is_slippery=slippery)
observation, info = env.reset()

def greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)  # OR np.argmax(qs)
    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}  #{0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
    action_probs[max_action] += (1 - epsilon)
    return action_probs

class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha
        
        self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)

agent = QLearningAgent()
state, _ = env.reset()
action = agent.get_action(state)
for _ in range(episodes):
    done = False
    state, _ = env.reset()
    while not done:
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_action = agent.get_action(next_state) if not done else None
        agent.update(state, action, reward, done)
        state = next_state
        action = next_action
    total_reward += reward
env.close()
print(f'total_reward:{total_reward}')