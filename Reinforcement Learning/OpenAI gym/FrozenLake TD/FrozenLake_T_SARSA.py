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

class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        self.pi = defaultdict(lambda: {0:0.25, 1:0.25, 2:0.25, 3:0.25})
        self.Q = defaultdict(lambda:0)
        self.memory = deque(maxlen=2) # 
    
    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return
        
        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        next_q = 0 if done else self.Q[next_state, next_action]

        target = reward + self.gamma * next_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)

agent = SarsaAgent()
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