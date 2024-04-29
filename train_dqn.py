from battle.envs import *
from battle.util import *
from battle.agents import *
from dqn import *
from torch.optim import RMSprop
import torch

initial_map = createN(10, 2)

dqn = DQN(network=QNetwork, optimizer=RMSprop, other_policies=(RandomAgent(),), grid=initial_map)

agent1, network = dqn.train(num_episodes=100)

torch.save(network.state_dict(), 'network.pth')

agent2 = RandomAgent()

env = GridBattle((agent1, agent2), initial_map)

env.run_game(100)

