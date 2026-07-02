# 初始化环境
import gymnasium as gym

env = gym.make('游戏名称-v1', render_mode='human') # 可视化

observation, info = env.reset()

print(f'{observation}')

done = False

while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    done = terminated or truncated

env.reset()
env.close()
