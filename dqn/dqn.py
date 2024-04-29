from .util import ReplayBuffer, Transition
from battle.util import createN, Tiles
from battle.agents import Agent, AgentFunction
from battle.envs import GridBattle
from numpy.typing import NDArray
from typing import Sequence, MutableSequence, Tuple
from functools import partialmethod
from collections.abc import Callable
from typing import Type
import math
import numpy as np
import torch
import torch.nn as nn
import gym


class DQN(object):
    DEFAULT_GRID_SHAPE: Sequence[int] | int = (40, 40)

    def __init__(self,
                 network: Type[nn.Module | Callable],
                 optimizer: Type[torch.optim.Optimizer],
                 optimizer_kwargs: dict | None = None,
                 num_copies: int = 1,
                 other_policies: Sequence[Agent] = tuple(),
                 grid: NDArray | None = None,
                 vision_window: Sequence[int] | int = (5, 5),
                 attack_window: Sequence[int] | int | None = None,
                 memory_capacity: int = 1000,
                 seed: int | None = None,
                 batch_size: int = 128,
                 gamma: float = 0.8,
                 eps_start: float = 0.95,
                 eps_end: float = 0.05,
                 eps_decay: float = 600,
                 target_update: int = 10,
                 ) -> None:
        """
        Sets up DQN with epsilon greedy policy, environment details, etc.

        :param network: Network to train
        :type network: Type[nn.Module | Callable]
        :param optimizer: Optimizer to use
        :type optimizer: Type[torch.optim.Optimizer]
        :param optimizer_kwargs: Kwargs to pass into optimizer
        :type optimizer_kwargs: dict | None
        :param num_copies: How many copies of itself to train against
        :type num_copies: int
        :param other_policies: Other policies to train against
        :type other_policies: Sequence[Agent]
        :param grid: The starting grid
        :type grid: NDArray | None
        :param vision_window: The window which each cell uses to calculate reward
        :type vision_window: Sequence[int, int] | int
        :param attack_window: The window in which each cell can attack
        :type attack_window: Sequence[int, int] | int | None
        :param memory_capacity: Capacity of experience replay buffer
        :type memory_capacity: int
        :param seed: The seed to send to the environment
        :type seed: int | None
        :param batch_size: Batch size
        :type batch_size: int
        :param gamma: Gamma for discounted sum of rewards
        :type gamma: float
        :param eps_start: Starting epsilon value for epsilon-greedy policy
        :type eps_start: float
        :param eps_end: Target epsilon value for epsilon-greedy policy
        :type eps_end: float
        :param eps_decay: Epsilon decay rate for epsilon-greedy policy
        :type eps_decay: float
        :param target_update: Frequency with which to update target network
        :type target_update: int
        """
        if grid is None:
            grid = createN(self.DEFAULT_GRID_SHAPE, len(other_policies) + 1)  # + 1 for our agent

        if isinstance(vision_window, int):
            vision_window: Tuple[int, int] = (vision_window, vision_window)
        if isinstance(attack_window, int):
            attack_window: Tuple[int, int] = (attack_window, attack_window)

        self.vision_window = vision_window  # H * W window of what a bot can see
        if attack_window is None:
            self.attack_window = self.vision_window
        else:
            self.attack_window = attack_window

        self.grid = grid
        self.seed = seed
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.n_attack_tiles = math.prod(self.attack_window)
        self.num_copies = num_copies
        self.total_agents = self.num_copies + len(other_policies)

        self.replay = ReplayBuffer(memory_capacity)
        self.timesteps = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.agents: MutableSequence = [None] * self.total_agents  # Networks will be randomly interspersed
        self.network = network(
            in_channels=self.total_agents + Tiles.AGENT,
            out_channels=self.n_attack_tiles,
            action_shape=(*self.attack_window, *self.grid.shape)
        ).to(self.device)
        self.target_network = network(
            in_channels=self.total_agents + Tiles.AGENT,
            out_channels=self.n_attack_tiles,
            action_shape=(*self.attack_window, *self.grid.shape)
        ).to(self.device)
        self.network_mask: MutableSequence = [None] * self.total_agents  # Mask for network agents
        if optimizer_kwargs is None:
            self.optimizer = optimizer(self.network.parameters())
        else:
            self.optimizer = optimizer(self.network.parameters(), **optimizer_kwargs)

        @AgentFunction
        def network_agent(obs: NDArray[int], action_space: gym.Space, obs_space: gym.Space) -> NDArray:
            return self.network(torch.tensor(obs, dtype=torch.float).unsqueeze(0)).squeeze(0).cpu().detach().numpy()

        self.network_agent = network_agent

        network_agent_ids = np.random.choice(self.total_agents, self.num_copies, replace=False)
        other_policy_idx = 0
        for i in range(self.total_agents):
            if i in network_agent_ids:
                self.agents[i] = self.network_agent(obs_onehot=True, attack_map=True, attack_map_logits=True)
                self.network_mask[i] = True
            else:
                self.agents[i] = other_policies[other_policy_idx]
                self.network_mask[i] = False
                other_policy_idx += 1

        self.agents: Sequence = tuple(self.agents)
        self.network_mask: Sequence[bool] = tuple(self.network_mask)


    def train(self, max_timestep: int = 200, num_episodes: int = 1000) -> Tuple[Agent, nn.Module]:
        env = GridBattle(self.agents, self.grid, self.vision_window, self.attack_window, temperature=0)
        print(f'Training for {num_episodes} episodes...')


        for i_episode in range(num_episodes):
            # Initialize the environment and state
            n_obs = env.reset(options={'max_timestep': max_timestep})
            timestep = 0
            total_rewards: Sequence[int] = [0] * self.total_agents
            done = False

            while not done:
                # Select and perform actions based on state, policies, and epsilon-greedy
                n_action: Sequence[NDArray] = self._select_actions(n_obs, env.action_spaces, env.observation_spaces)

                n_next_obs, n_reward, done, info = env.step(n_action)

                # Store the transition in memory for each copy of the agent
                for i, (obs, action, reward, next_obs, agent, is_network) in \
                        enumerate(zip(n_obs, n_action, n_reward, n_next_obs, self.agents, self.network_mask)):
                    if is_network:  # If this is a network policy (policy we are training)
                        if next_obs[Tiles.AGENT].sum() > 0:  # If at least one cell from this agent is alive
                            total_rewards[i] += reward.sum()
                            self.replay.push(obs, action, reward, next_obs)

                # Move on to the next state
                n_obs = n_next_obs

                # Perform one step of the optimization (on the policy network)
                self._optimize_network()

                timestep += 1
            # Update the target network, copying all weights and biases in our main network
            if i_episode % self.target_update == 0:
                self.target_network.load_state_dict(self.network.state_dict())

            avg_total_reward = sum(total_rewards) / self.num_copies
            print(f'Episode {i_episode:4} | Game timesteps: {timestep:4} | Average Total Reward: {avg_total_reward}')

        print('Finished training!')
        return self.network_agent(obs_onehot=True, attack_map=True, attack_map_logits=True), self.network


    def _optimize_network(self) -> None:
        if len(self.replay) < self.batch_size:
            return

        transitions = self.replay.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions to Transition of batch-arrays
        # This creates a transition with a list for each element (current state, current action, ...)
        batch = Transition(*zip(*transitions))

        # tensor of all states (N, C, H_g, W_g)
        state_batch = torch.tensor(np.stack(batch.state), dtype=torch.float, device=self.device)
        # tensor of all actions (N, H_w, W_w, H_g, W_g)
        action_batch = torch.tensor(np.stack(batch.action), dtype=torch.float, device=self.device)
        # tensor of all rewards (N, H_g, W_g)
        reward_batch = torch.tensor(np.stack(batch.reward), dtype=torch.float, device=self.device)
        next_state_batch = torch.tensor(np.stack(batch.next_state), dtype=torch.float, device=self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # self.network(state_batch) is (N, H_w, W_w, H_g, W_g)
        # action_batch is (N, H_w, W_w, H_g, W_g)
        # Attack window of action batch is flattened, so (N, D, H_g, W_g)
        flattened_action_batch = action_batch.flatten(start_dim=1, end_dim=2)
        # Get highest Q values, so (N, 1, H_g, W_g)
        max_q_batch = flattened_action_batch.amax(1, keepdims=True)
        # (N, H_w, W_w, H_g, W_g) -> (N, D, H_g, W_g)
        network_output = self.network(state_batch).flatten(start_dim=1, end_dim=2)
        # After indexing for action taken, state_action_values are 0 along D dimension except for max
        state_action_values = network_output * (max_q_batch == network_output)
        # Now we can sum across D dimension (reducing it) to get our final Q's (N, H_g, W_g)
        state_action_values = state_action_values.sum(dim=1)

        # Compute V(s_{t+1}) for all next states.
        # (N, H_w, W_w, H_g, W_g) -> (N, H_g, W_g) [H_w, W_w dims get removed because we are getting max(Q)]
        next_state_values = self.target_network(next_state_batch).amax(dim=(1, 2)).detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def _select_actions(self,
                        n_obs: Sequence[NDArray],
                        n_action_space: Sequence[gym.Space],
                        n_obs_space: Sequence[gym.Space]
                        ) -> Sequence[NDArray]:
        """
        Select agent actions based on epsilon-greedy policy. n_obs should be numpy arrays from environment, and it will
        return the actions performed by each agent

        :param n_obs: Observations to select actions from
        :type n_obs: Sequence[NDArray]
        :return: Selected actions
        :rtype: NDArray
        """
        assert len(n_obs) == len(self.agents), "States sent to _select_action don't match number of initialized agents!"

        current_epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.timesteps / self.eps_decay)
        self.timesteps += 1

        n_action: MutableSequence = [None] * self.total_agents
        for i, (obs, action_space, obs_space, agent, is_network) in \
                enumerate(zip(n_obs, n_action_space, n_obs_space, self.agents, self.network_mask)):
            if not is_network:  # If this is a non-network policy (policy we are training against)
                n_action[i] = agent.policy(obs, action_space, obs_space)
                continue

            # Random actions is (H_w, W_w, H_g, W_g)
            random_actions = np.random.randint(0, self.n_attack_tiles, (*self.attack_window, *self.grid.shape))

            # Random mask is (H_g, W_g)
            random_mask = np.random.binomial(1, 1 - current_epsilon, self.grid.shape)  # Mask based on epsilon

            with torch.no_grad():
                action = agent.policy(obs, action_space, obs_space)
            # Action is (H_w, W_w, H_g, W_g)
            action[random_mask] = random_actions[random_mask]

            n_action[i] = action

        return n_action
