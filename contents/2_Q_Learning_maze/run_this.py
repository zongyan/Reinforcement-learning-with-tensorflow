"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset() # 每一次进入一个新的episode，都是需要充重置一下环境的了，让其回到初始的状态。
                                  # 原来observation返回的就是第一个例子中的状态了 
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            # 其实就是和第一个例子里面是类似的，就是返回state，reward，terminal
            # 我的理解，就是根据action，选择进入了下一个状态(observation)，
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

# 这个文档不同之处，就是在于
if __name__ == "__main__":
    env = Maze() # 初始化环境
    RL = QLearningTable(actions=list(range(env.n_actions))) # 初始化Q Table

    env.after(100, update)
    env.mainloop() # 会调用update函数，但是现在需要搞清楚一下，这个gym到底是怎么运行的了。