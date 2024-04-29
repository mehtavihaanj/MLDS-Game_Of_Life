from numpy.typing import NDArray
from battle.envs import GridBattle
from battle.util import createN, create1v1, Tiles, sample_map_1_sliding
from battle.agents import Agent, AgentFunction


@AgentFunction
def myAgent(obs, action_space, obs_space) -> NDArray:
    return action_space.sample()


agent1 = myAgent()

env = GridBattle((agent1,) * 2, sample_map_1_sliding)

env.run_game(200)
