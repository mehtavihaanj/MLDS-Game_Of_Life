import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from battle.envs import ParallelGridBattleRL
from battle.util import Tiles


class Actor(nn.Module):
    def __init__(self, env: ParallelGridBattleRL):  # use env to get in/out shapes
        super(Actor, self).__init__()
        self.in_shape = (env.n_agents + Tiles.AGENT, *env.vision_window)
        self.out_shape = env.attack_window  # H * W probability distribution of where to attack

        self.conv1 = nn.Conv2d(in_channels=self.in_shape[0], out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(size=self.out_shape)
        self.bn2 = nn.BatchNorm2d(num_features=512)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        # Convert observation to tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        x = self.bn1(F.relu(self.pool1(self.conv1(x))))
        x = self.bn2(F.relu(self.upsample1(self.conv2(x))))
        x = torch.squeeze(self.conv3(x), dim=1)  # squeeze the channel dimension out

        return x  # (N, H, W)


class Critic(nn.Module):
    def __init__(self, env: ParallelGridBattleRL):
        super(Critic, self).__init__()
        self.in_shape = (env.n_agents + Tiles.AGENT, *env.vision_window)
        self.out_shape = env.attack_window  # H * W probability distribution of where to attack

        self.conv1 = nn.Conv2d(in_channels=self.in_shape[0], out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=512)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.bn3 = nn.BatchNorm1d(num_features=2048)
        self.fc1 = nn.Linear(in_features=2048, out_features=256)
        self.bn4 = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        # Convert observation to tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        x = self.bn1(F.relu(self.pool1(self.conv1(x))))
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x))).flatten(1)  # flatten all dimensions except batch
        x = self.bn3(x)
        x = self.bn4(F.relu(self.fc1(x)))
        x = self.fc2(x).squeeze()  # just batch dimension of critic scores for each sample in batch

        return x
