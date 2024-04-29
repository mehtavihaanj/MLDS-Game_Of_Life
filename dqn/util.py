import random
from collections import namedtuple, deque
import torch
import torch.nn as nn

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


# Replay buffer to implement Experience Replay
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)  # Internal data structure for the buffer

    def push(self, *args, **kwargs):
        """Save a transition"""
        self.memory.append(Transition(*args, **kwargs))  # Adds a transition to the buffer

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # Randomly samples from the buffer

    def __len__(self):
        return len(self.memory)  # Number of elements in the buffer


# torch.Tensor.View as a module so that it can be used in torch.Sequential
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        if len(x.shape) == 3:  # If (C, H, W)
            return x.view(*self.shape)
        else:   # If (*, C, H, W)
            return x.view(*x.shape[:-3], *self.shape)


# torch.Permute as a module so that it can be used in torch.Sequential
class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)
