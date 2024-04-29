import gym
import numpy as np
from battle.envs import *
from battle.util import *
env = ParallelGridBattleRL(use_logits=True)
n_obs = env.reset()
done = False

while not done:
    env.render()

    action = env.action_space.sample()

    n_obs, n_reward, done, info = env.step(action)

env.close()
