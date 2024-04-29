import gym
import numpy as np
from battle.envs import *
from battle.util import *
from ppo import PPO, Actor, Critic
import torch

env = ParallelGridBattleRL(use_logits=True)

model = PPO(env)
# model = PPO(env, actor='actor.pth', critic='critic.pth')

model.learn(iterations=200_000_000)
