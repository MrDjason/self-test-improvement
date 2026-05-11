import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
LAMBDA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3

# 建立Q表

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns = actions
    )
    print(table)
    return table

# 选择行动

def choose_action(state, q_table):
    state_action =  q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or state_action.all() == 0: 
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_action.argmax()
    return action_name

# 更新环境

def update_env(S, episode, step_counter):
    env_list = ['-']*(N_STATES-1) + ['T']
    if S == 'terminal':
        interaction = f'Episode:{episode+1}, total_step:{step_counter}'
        print(f'\r{interaction}', end='')
        time.sleep(2)
        print('\r                          ', end='')
    else:
        env_list[S] = 'O'
        interaction = ''.join(env_list)
        time.sleep(FRESH_TIME)

# 获取反馈

def get_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2:
            S = 'terminal'
            R = 1
        else:
            S += 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ -= 1
    return S_, R

# 强化学习

def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_feedback(S, A)
            q_predict = q_table.iloc[S, A]
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            
            q_table.iloc[S, A] += ALPHA * (q_target - q_predict)
            S = S_

            update_env(S, episode, step_counter+1)
            step_counter += 1
        return q_table

if __name__ == '__main__':
    q_table = rl()
    print(q_table)