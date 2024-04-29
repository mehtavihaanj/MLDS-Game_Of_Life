from battle.agents import Agent
from numpy.typing import NDArray
import gym


class RandomAgent(Agent):
    def __init__(self,
                 obs_onehot: bool = False,
                 attack_map: bool = False,
                 attack_map_logits: bool = False,
                 ):
        super().__init__(obs_onehot, attack_map, attack_map_logits)

    def policy(self, obs: NDArray, action_space: gym.Space, obs_space: gym.Space) -> NDArray:
        return action_space.sample()
