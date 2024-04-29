from numpy.typing import NDArray
from typing import Callable, Type
import abc
import gym


class Agent(object):
    __metaclass__ = abc.ABCMeta


    def __init__(self,
                 obs_onehot: bool = False,
                 attack_map: bool = False,
                 attack_map_logits: bool = False,
                 ):
        """
        :param obs_onehot: Determines whether the observation that is received is one-hot encoded or not
        :type obs_onehot: bool
        :param attack_map: Determines whether you pass in a probability map for each cell's attack
        (4 dimensional array for your total action)
        :type attack_map: bool
        :param attack_map_logits: If you pass in a map for each cell's attack, this determines whether it
        is a logit or probability map
        :type attack_map_logits: bool
        """
        self.obs_onehot = obs_onehot
        self.attack_map = attack_map
        self.attack_map_logits = attack_map_logits

    @abc.abstractmethod
    def policy(self, obs: NDArray[int], action_space: gym.Space, obs_space: gym.Space) -> NDArray:
        raise NotImplementedError("Please implement the policy method!")

    __call__ = policy


def AgentFunction(fn: Callable[[NDArray[int], gym.Space, gym.Space], NDArray]) -> Type[Agent]:
    class CallableAgent(Agent):
        def policy(self, obs: NDArray[int], action_space: gym.Space, obs_space: gym.Space) -> NDArray:
            return fn(obs, action_space, obs_space)
    return CallableAgent
