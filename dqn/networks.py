from functools import partial
import gym
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .util import View, Permute
from typing import Sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QNetwork(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 action_shape: Sequence,
                 ):
        super(QNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding='same', padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same', padding_mode='circular')
        # self.bn2 = nn.BatchNorm2d(16)
        # self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same', padding_mode='circular')
        # self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding='same', padding_mode='circular')
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.view1 = View(action_shape)

        # End with the same amt of outputs as there are choices to attack (3*3, 5*5, etc.)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        return self.view1(F.relu(self.bn4(self.conv4(x))))
