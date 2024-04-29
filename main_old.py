import gym
import numpy as np
from battle.envs import *
from battle.util import *
from time import perf_counter

map = create1v1(10)
# map[4:6] = Tiles.WALL
# map[4:6, 4:6] = Tiles.EMPTY

env = GridBattleHuman(initial_map=map)
n_obs = env.reset()
done = False


def produce_probability_map(cell_obs):
    attack_tiles = cell_obs[0] + cell_obs[3]
    total = attack_tiles.sum()
    if total <= 0:
        return cell_obs[2] / cell_obs[2].sum()
    else:
        return attack_tiles / total


def produce_agent_action(obs):
    action = [produce_probability_map(cell_obs) for cell_obs in obs]
    return np.array(action)


t1_start = perf_counter()

while not done:
    env.render()

    n_action = [produce_agent_action(obs) for obs in n_obs]

    n_obs, n_reward, done, info = env.step(n_action)

t1_stop = perf_counter()
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)

env.close()
