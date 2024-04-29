import gym
from battle.envs import *
from battle.util import *
from ppo import Actor
import numpy as np
import torch
import torch.nn.functional as F


np.random.seed(1)
torch.manual_seed(1)

map = create1v1(10)
env = ParallelGridBattleRL(initial_map=map, use_logits=False)

actor = Actor(env).to('cuda')
actor_file = torch.load('actor.pth')
actor.load_state_dict(actor_file['state_dict'])


def produce_probability_map(cell_obs):
    attack_tiles = cell_obs[0] + cell_obs[3]
    total = attack_tiles.sum()
    if total <= 0:
        return cell_obs[2] / cell_obs[2].sum()
    else:
        return attack_tiles / total

    # a = np.zeros((5, 5))
    # a[0, 0] = 1
    #
    # return a


n_obs, masks = env.reset()
done = False

while not done:
    env.render()

    n_action = env.action_space.sample()
    # n_action = np.zeros(shape=env.action_space.shape)
    # n_action[:, 2, 2] = 1  # default attack

    # agents with id 3 is us and id 4 is enemy
    agent_mask, enemy_mask = masks

    # tensorize n_obs for processing
    n_obs = torch.from_numpy(n_obs).float().to('cuda')

    agent_logits = actor(n_obs)
    action_shape = agent_logits.shape

    flattened_logits = agent_logits.view((agent_logits.shape[0], -1)).detach().cpu()
    flattened_action = F.softmax(flattened_logits, dim=1)
    agent_action = flattened_action.view(action_shape).numpy()

    # set agents 3's actions
    n_action[enemy_mask == False] = agent_action[enemy_mask == False]

    # -- agents with id 4 will do a default action --
    pass

    # -- agents with id 4 will attack a random valid tile --
    # for i, cell_obs in enumerate(n_obs):
    #     if not enemy_mask[i]:
    #         continue
    #     n_action[i] = produce_probability_map(cell_obs).detach().cpu().numpy()

    for i, cell_action in enumerate(n_action):  # dumb floating point inaccuracy >:(
        n_action[i] /= np.sum(cell_action)

    n_full_obs, n_reward, done, info = env.step(n_action)
    n_obs, masks = n_full_obs


env.close()
