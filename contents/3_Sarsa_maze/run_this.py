"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from maze_env import Maze
from RL_brain import SarsaTable


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()

# 其实，line 20 - line 30，就是可以看出这个sarsa&q learning的区别的了。对于q learning
# 来说，因为他仅仅是需要基于现有的状态（即observation），然后获取一个action，接着就是可以
# 知道相应的下一个状态，就是根据sars更新q table的了。
# 这个就是两种算法的不同点，对于sarasa来说，出来需要知道现在的状态，以及这个状态下的reward，
# action之外，还是需要知道下一个状态的对应的action，从而就是可以使用这一些的内容进行更新的了。

# 当然了，以上这一些都是从更新的逻辑上面来看的了，如果是从RLBrain层面来看，主要就似乎在于Q 
# learning使用的是max获取Q target，而SARSA的方式，则是通过最后一个A（Action）获取Q Target的了

