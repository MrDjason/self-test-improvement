import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

env = gym.make('Blackjack-v1', natural=False, sab=False)
observation, info = env.reset()

total_reward = 0
episodes = []
rewards = []
# 一般性策略
for i in range(100):
    observation, info = env.reset()
    done = False
    while not done:
        print(f'决策前：玩家总点数{observation[0]}，庄家一点数{observation[1]}，是否有A{observation[2]}') # 玩家总点数、庄一牌点数、是否有A

        action = env.action_space.sample() # 随机选择要牌或者不要牌

        observation, reward, terminated, truncated, info =  env.step(action)
        print(f'决策后：玩家总点数{observation[0]}，庄家一点数{observation[1]}，是否有A{observation[2]}') # 玩家总点数、庄一牌点数、是否有A

        total_reward += reward
        done = terminated or truncated

    episodes.append(i+1)
    rewards.append(total_reward)

    env.reset()



print(f'总奖励点数：{total_reward}')
env.close()

plt.plot(episodes, rewards)
plt.xlabel('局数')
plt.ylabel('总奖励')
plt.title('随机策略')
plt.show()