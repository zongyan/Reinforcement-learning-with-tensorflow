"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible


N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))), # 初始化Q table 
        columns=actions,    # 定义column的名字
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS) # 这里就是直接是使用全局变量的形式进行ramdom选择了。
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1 # 那这里的含义，就是每一次执行一个动作，环境仅仅是会向左，或者向右移动一步的吗？
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table) # 根据state S，以及现在Q table数值，选择一个action（greedy）
            S_, R = get_env_feedback(S, A)  # 执行动作，获得下一个状态和reward
            q_predict = q_table.loc[S, A] # 获取这个状态&动作下，此时的Q value， 之所以是predict的原因，因为Q
                                          # 表格就是一直是predict的数值了 
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal， 注意一下，这里是下一个状态的Q value
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table

"""
看了这段代码之后，都这个Q Learning有了一个基本的了解，对于代码的实现，还是相对来说很简单的：

1. 根据状态&action的个数，初始化Q table
2. 更新环境
3. 调用一个函数，从而选择一个action，并且返回这个action (对于agent来说的)
4. 根据现在所处的状态&选择的action，获得下一个的状态，以及相应的reward（对于环境来说的）
5. 根据所处的状态&选择的action，获得Q值 [q_predict]（需要注意的是，这个Q Table里面的数值，都是predict的结果，和环境没有任何的关系）
6. 计算出一个新的Q值 [q_target]，这个Q target，是下一个状态的Q value
7. 更新Q Table （更新的还是本状态下的Q value）
8. 移动到下一个状态
9. 更新环境

大概的过程我算是理解的了，但是就是对于这个环境的更新，我暂时还是不太明白，到底是一个怎么样子的更新？因为这个是我非常关注的一个东西的了。
不过不要着急，这个就是在后期肯定就是会自然而然的明白的了。

"""



if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
