from gym import spaces
import numpy as np
from typing import Tuple, Union
from numpy.typing import NDArray
from collections.abc import MutableSequence, Sequence, Iterable
from typeguard import check_argument_types
import torch.nn.functional as F
import torch
from ..util import *
from ..agents import Agent, RandomAgent
import pygame
import gym
import math
from scipy import signal


class GridBattle:
    """
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 2,
    }

    DEFAULT_MAX_TIMESTEP = 200
    TILE_COLORS = [  # colors to assign to each tile type when displaying
        (255, 255, 255),  # EMPTY
        (50, 50, 50),  # WALL
        (255, 0, 0),  # FRIENDLY AGENT
        (0, 0, 255),  # ENEMY AGENT 1
        (0, 255, 0),  # ENEMY AGENT 2
        (255, 255, 0),  # etc.
        (0, 255, 255),
        (255, 0, 255),
        (255, 127, 0),
        (127, 255, 0),
        (0, 127, 255),
        (255, 0, 127),
    ]
    KEYS_TO_AGENTS = {  # mappings from keys to relative agents ids
        pygame.K_0: 0,
        pygame.K_1: 1,
        pygame.K_2: 2,
        pygame.K_3: 3,
        pygame.K_4: 4,
        pygame.K_5: 5,
        pygame.K_6: 6,
        pygame.K_7: 7,
        pygame.K_8: 8,
        pygame.K_9: 9,
    }
    OVERPOPULATION_THRESHOLD = 6  # if a cell has more than this many neighbors, die

    # action and observation spaces for each agent as well as environment seed
    action_spaces: Sequence[gym.Space]
    observation_spaces: Sequence[gym.Space]
    seed: int

    def __init__(self,
                 agents: Sequence[Agent] = (RandomAgent, RandomAgent),
                 initial_map: Union[NDArray[int], Sequence[int, int, int], Sequence[int, int], int] = 8,
                 vision_window: Union[Sequence[int, int], int] = (5, 5),
                 attack_window: Union[Sequence[int, int], int, None] = None,
                 temperature: float = 1,
                 window_height: int = 560,
                 seed: int = None,
                 ) -> None:
        """
        Create the grid battle environment. The vision window determines the H and W axes of the observation space,
        which will not change in the future.

        :param agents: A set of agents to perform their policies in the environment
        :type agents: Sequence[Agent]
        :param initial_map: An initial map (T * H * W enum-encoded numpy array), a height/width tuple, or a side length int
        :type initial_map: Union[NDArray[int], Sequence[int, int, int], Sequence[int, int], int]
        :param vision_window: A height/width tuple or side length int that sets cells' H * W vision window dimensions
        :type vision_window: Union[Sequence[int, int], int]
        :param attack_window: An optional height/width tuple or side length int for cells' attack window dimensions
        :type attack_window: Union[Sequence[int, int], int, None]
        :param temperature: The temperature to sample logit maps at.
        :type temperature: float
        :param window_height: The height of the rendering window in pixels
        :type window_height: int
        :param seed: Seed, only affects observation and action spaces before reset()
        :type seed: int
        """
        if isinstance(initial_map, Sequence):
            initial_map: NDArray[int] = np.array(initial_map, dtype=np.ushort)
        elif isinstance(initial_map, int):
            initial_map: NDArray[int] = create1v1(initial_map)
        
        if len(initial_map.shape) == 2:  # if there's no time dimension
            initial_map = np.expand_dims(initial_map, axis=0)

        if isinstance(vision_window, int):
            vision_window: Tuple[int, int] = (vision_window, vision_window)
        if isinstance(attack_window, int):
            attack_window: Tuple[int, int] = (attack_window, attack_window)

        self.vision_window = vision_window  # H * W window of what a bot can see
        if attack_window is None:
            self.attack_window = self.vision_window
        else:
            self.attack_window = attack_window
        self.initial_map = np.copy(initial_map)  # store a copy of this that we can reset to in the future
        self.map_time_length: int = initial_map.shape[0]  # number of time steps in the map
        self.shape: Tuple[int, int] = initial_map.shape[1:]  # grid shape (H_g, W_g)
        self.max_timestep = self.DEFAULT_MAX_TIMESTEP

        # for displaying in a human viewable format
        self.tile_height = window_height // self.shape[0]
        self.tile_size = (self.tile_height, self.tile_height)  # size of each tile in pixels
        height = self.shape[0] * self.tile_height
        width = self.tile_height * self.shape[1]
        self.window_size = (height, width)

        self.temperature = temperature
        self.timestep = 0

        self.grid: NDArray[int] = np.empty(self.shape, dtype=np.byte)
        self.agents: Sequence[Agent] = agents
        self.n_tile_types = np.amax(initial_map) + 1
        self.n_agents: int = self.n_tile_types - Tiles.AGENT  # number of agents determined by initial map tile values
        self.n_attack_tiles = math.prod(self.attack_window)

        assert self.n_agents == len(self.agents), '# of agents in map must agree with # of agent objects passed in'

        if self.n_agents + Tiles.AGENT > len(self.TILE_COLORS):
            for i in range(self.n_agents + Tiles.AGENT - len(self.TILE_COLORS)):
                self.TILE_COLORS.append(tuple(np.random.choice(range(256), size=3).tolist()))

        self._set_env_spaces(seed)

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct frame rate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs_space(self, agent: Agent, seed: int = 0) -> gym.Space:
        if not agent.obs_onehot:  # if we are using enum encoding
            return spaces.MultiDiscrete(np.full(self.shape, self.n_tile_types), seed=seed)

        # if we are using one-hot encoding
        return spaces.MultiBinary((self.n_tile_types, *self.shape), seed=seed)

    def _get_action_space(self, agent: Agent, seed: int = 0) -> gym.Space:
        if not agent.attack_map:  # if we are using raw integers
            return spaces.MultiDiscrete(np.full(self.shape, self.n_attack_tiles), seed=seed)

        if not agent.attack_map_logits:  # if we are using probability maps
            return spaces.Box(low=0.0, high=1.0, shape=(*self.shape, *self.attack_window), seed=seed)

        # if we are using logit maps
        return spaces.Box(low=-np.inf, high=np.inf, shape=(*self.shape, *self.attack_window), seed=seed)

    def _get_obs(self) -> Sequence[NDArray]:
        """
        Gets a full observation. This should only need to be called once per step, it provides a copy of the whole grid
        from the perspective of each agent.

        :return: A sequence of NDArray observations for each agent.
        :rtype: Sequence[NDArray]
        """
        agent_observations: MutableSequence[Union[NDArray, None]] = [None] * self.n_agents
        global_agent_mask: NDArray = self.grid == Tiles.AGENT  # mask of cells belonging to agents Tiles.AGENT
        for rel_agent_id, agent in enumerate(self.agents):
            agent_id = rel_agent_id + Tiles.AGENT  # relative agent id -> de facto agent id

            agent_mask: NDArray = self.grid == agent_id  # mask of cells belonging to agents agent_id

            perspective_grid = np.copy(self.grid)  # a copy with our agent id swapped with agent of id Cells.AGENT
            perspective_grid[global_agent_mask] = agent_id
            perspective_grid[agent_mask] = Tiles.AGENT

            if not agent.obs_onehot:  # if we are good with this enum encoded representation
                agent_observations[rel_agent_id] = perspective_grid

                continue

            # if we wanna onehot encode this
            agent_observation = np.zeros((self.n_tile_types, *self.shape), dtype=bool)

            # https://stackoverflow.com/questions/63839024/one-hot-encode-numpy-array-with-2-dims
            agent_observation[(perspective_grid, *np.indices(self.shape, sparse=True))] = True

            agent_observations[rel_agent_id] = agent_observation

        return agent_observations

    def _get_info(self) -> dict:
        return {
            'grid': self.grid
        }

    def _get_value_maps(self, reward_kernel: NDArray) -> NDArray:
        value_maps: NDArray = np.empty(shape=(self.n_agents, *self.grid.shape))
        for rel_agent_id in range(self.n_agents):
            agent_id = rel_agent_id + Tiles.AGENT  # relative agent id -> de facto agent id

            enemy_grid = np.ones_like(self.grid, dtype=np.byte) - (self.grid == Tiles.EMPTY) - (self.grid == Tiles.WALL) - (self.grid == agent_id)
            friendly_grid = self.grid == agent_id
            rel_reward_grid = friendly_grid - enemy_grid

            value_maps[rel_agent_id] = signal.convolve2d(rel_reward_grid, reward_kernel, boundary='wrap', mode='same')

        return value_maps

    # checks if a 2d point is a valid point to attack
    def _valid_cell(self, pos: Union[Iterable, Sequence[int, int], NDArray[int]]) -> bool:
        # for axis, axis_length in enumerate(self.grid.shape):
        #     if not 0 <= pos[axis] < axis_length:  # if out of bounds in some axis
        #         return False

        return self.grid[tuple(pos)] != Tiles.WALL

    def _set_env_spaces(self, seed: Union[int, None]) -> None:
        # the shape of each agent's action
        self.action_spaces: Sequence[gym.Space] = tuple(self._get_action_space(agent, seed) for agent in self.agents)

        # the shape of each agent's observation
        self.observation_spaces: Sequence[gym.Space] = tuple(self._get_obs_space(agent, seed) for agent in self.agents)

    def _normalize_actions(self, n_action: Sequence[NDArray]) -> Sequence[Union[NDArray, None]]:
        # given agent actions n_action, convert it from a sequence of agent actions in varying formats
        # (logits, probability maps, 0-24 coordinates) to pairs of coordinates
        normalized_actions: MutableSequence[Union[NDArray, None]] = [None] * self.n_agents

        for rel_agent_id, (agent_action, agent) in enumerate(zip(n_action, self.agents)):
            logit_maps = agent_action.astype(np.double)

            if agent.attack_map and agent.attack_map_logits:  # if using logits
                # pytorch can only take softmax over one dimension at a time, so we must flatten first 2 dims
                flattened_logits = logit_maps.reshape((-1, *logit_maps.shape[2:]))
                if self.temperature == 0:  # temperature == 0 means to just one-hot encode highest value
                    flattened_probs = flattened_logits == flattened_logits.max(axis=0)
                else:
                    flattened_probs = F.softmax(torch.Tensor(flattened_logits/self.temperature), dim=0).detach().numpy()

                probability_maps = flattened_probs.reshape(logit_maps.shape)
            else:
                probability_maps = logit_maps

            if agent.attack_map:  # if using probability map or logits
                # normalize probability maps (in case of floating point errors or something)
                action = probability_maps.astype(np.longdouble)
                action_sum = np.broadcast_to(action.sum(axis=(0, 1)), action.shape)  # sums of the probability maps
                valid_action_mask = action_sum > 0
                random_action_mask = action_sum == 0

                # normalize by dividing each probability map by their sums (only for non-zero sums)
                action[valid_action_mask] = action[valid_action_mask] / action_sum[valid_action_mask]

                if random_action_mask.sum() > 0:  # probability maps that sum to 0 are set to be uniformly distributed
                    action[random_action_mask] = np.full_like(action[random_action_mask], 1 / self.n_attack_tiles)

                # flatten probability maps so that we can sample (H_w * W_w * H_g * W_g) -> (D * H_g * W_g)
                flattened_action = action.reshape((-1, *agent_action.shape[2:]))

                index_maps = np.empty(action.shape[2:], dtype=np.uintc)

                # for some reason numpy doesn't offer a fast way to sample multiple probability distributions
                # simultaneously, so time to go with good old slow python
                for i in range(flattened_action.shape[1]):
                    for j in range(flattened_action.shape[2]):
                        distribution = flattened_action[..., i, j]
                        # sample an index from the 1d array with the probability distribution from the original 2d array
                        sample_index = np.random.choice(a=distribution.size, p=distribution)
                        index_maps[i, j] = sample_index
            else:
                index_maps = probability_maps

            # if we are using raw integers
            # convert integers to coordinates, this is hardcoded for 2 dimensions (can u make it n-dimensional? :0)
            normalized_actions[rel_agent_id] = np.broadcast_to(index_maps, (2, *agent_action.shape[-2:]))\
                .astype(dtype=np.ushort)
            normalized_actions[rel_agent_id] = np.moveaxis(normalized_actions[rel_agent_id], 0, -1)
            normalized_actions[rel_agent_id][..., 0] //= self.attack_window[0]
            normalized_actions[rel_agent_id][..., 1] %= self.attack_window[1]

        return normalized_actions

    def _get_attacked_grid(self, normalized_actions: Sequence[Union[NDArray, None]]):
        # grid that shows how much each tile is being attacked by each agent
        attacked_grid = np.zeros((*self.shape, self.n_agents), dtype=np.ushort)

        # first, sample each cell's attack probability distribution
        # iterate in the same way that the observations are iterated over, so we have the same indices
        attack_offset = np.array(tuple((attack_size_axis - 1) // 2 for attack_size_axis in self.attack_window))

        for rel_agent_id in range(self.n_agents):
            agent_id = rel_agent_id + Tiles.AGENT  # relative agent id -> de facto agent id

            for cell_axis0, cell_axis1 in zip(*np.where(self.grid == agent_id)):
                cell_pos = np.array([cell_axis0, cell_axis1])
                attack_pos = ((cell_pos + normalized_actions[rel_agent_id][cell_axis0, cell_axis1] - attack_offset) % np.array(self.shape)).astype(np.uint)
                if self._valid_cell(attack_pos):
                    attacked_grid[tuple(attack_pos)][rel_agent_id] += 1  # indicate the tile is being attacked

        # by now the attacked_grid should be populated with the tiles that are attacked by each agent
        return attacked_grid  # (H, W, N)

    def reset(self,
              *,
              seed: Union[int, None] = None,
              return_info: bool = False,
              options: Union[dict, None] = None,
              ) -> Union[Tuple[Sequence[NDArray], dict], Sequence[NDArray]]:
        self.grid = np.copy(self.initial_map[0])  # reset grid
        self.seed = seed
        self.timestep = 0

        self._set_env_spaces(seed)

        if not self.max_timestep:
            self.max_timestep = options['max_timestep'] if isinstance(options, dict) and 'max_timestep' in options \
                else self.DEFAULT_MAX_TIMESTEP

        n_obs = self._get_obs()
        info = self._get_info()
        return (n_obs, info) if return_info else n_obs

    def step(self, n_action: Sequence[NDArray]) -> Tuple[Sequence[NDArray], NDArray, bool, dict]:
        """
        The rewards are given on the basis of cells meaning each agent gets a grid of cell rewards. The reward is
        calculated as enemy cells - friendly cells in the vision area of each cell.

        :param n_action: A list of actions for each agent to take, where each agent's actions is in its action space.
        :type n_action: Sequence[NDArray]
        :return: A list of observations, a list of reward maps, whether the environment is done or not, and information.
        :rtype: Tuple[Sequence[NDArray], Sequence[NDArray], bool, dict]
        """
        assert check_argument_types(), \
            'action must be an NDArray'

        # assert number of actions corresponds with number of agents
        assert len(n_action) == self.n_agents, 'number of actions is different from number of agents!'

        normalized_actions: Sequence[Union[NDArray, None]] = self._normalize_actions(n_action)
        # now, whether we were using logit maps, probability maps, or integers, they should be maps of coordinates

        # maps for calculating reward maps
        reward_kernel = np.ones(self.vision_window)
        pre_step_value_maps: NDArray = self._get_value_maps(reward_kernel)

        # update timestep and whether we are done or not
        self.timestep += 1
        done = self.timestep >= self.max_timestep  # end if time has run out

        ### MAP UPDATE PHASE ###
        if self.map_time_length > 1:  # if we have a map that changes over time
            relative_timestep: int = self.timestep % self.map_time_length
            new_changes: NDArray = self.initial_map[relative_timestep]  # get the changes for this timestep
            new_changes_mask: NDArray = new_changes != Tiles.EMPTY  # mask of cells that are changing
            removal_mask: NDArray = (self.grid == Tiles.WALL) & (new_changes == Tiles.EMPTY)  # mask of walls that are being removed
            self.grid[new_changes_mask] = new_changes[new_changes_mask]  # update the grid with the changes
            self.grid[removal_mask] = Tiles.EMPTY  # remove walls that are being removed

        ### ATTACK/CONVERSION PHASE ###
        # grid that shows how much each tile is being attacked by each agent
        attacked_grid = self._get_attacked_grid(normalized_actions)

        # a grid that says the total times each tile was attacked by any team
        total_attacked_grid = attacked_grid.sum(axis=-1, initial=0)

        converted_mask = total_attacked_grid > 0  # only convert the tiles that were attacked at least once. also avoid division by 0 error
        # basically just copy the converted mask over the agent dimension so we can use it for indexing attacked_grid
        converted_mask_expanded = np.broadcast_to(np.expand_dims(converted_mask, axis=-1), (*converted_mask.shape, attacked_grid.shape[-1]))

        # for each attacked tile, sample from the distribution of agent attacks (risk rules)
        attacked_tiles = attacked_grid[converted_mask_expanded]  # get the attacked tiles
        attacked_tiles = attacked_tiles.reshape((-1, self.n_agents))  # unflatten the attacked tiles to be an array of attack distributions
        total_attacked_tiles = total_attacked_grid[converted_mask]  # get the total number of times each tile was attacked
        attacked_tiles = attacked_tiles / np.expand_dims(total_attacked_tiles, -1)  # normalize the attacked tiles
        converted_tiles = np.apply_along_axis(lambda distribution: np.random.choice(a=self.n_agents, p=distribution) + Tiles.AGENT, axis=1, arr=attacked_tiles)

        new_grid = np.copy(self.grid)
        new_grid[converted_mask] = converted_tiles.astype(np.ubyte)

        self.grid = new_grid

        ### DEATH PHASE ###
        # punishment for overpopulation
        overpopulation_kernel = np.ones((3, 3))
        living_cell_grid = self.grid >= Tiles.AGENT
        neighbor_count_grid = signal.convolve2d(living_cell_grid, overpopulation_kernel, boundary='wrap', mode='same')
        overpopulated_mask = neighbor_count_grid > self.OVERPOPULATION_THRESHOLD
        self.grid[overpopulated_mask * living_cell_grid] = Tiles.EMPTY  # kill living overpopulated cells

        # maps for calculating reward maps
        post_step_value_maps: NDArray = self._get_value_maps(reward_kernel)
        # reward is the difference in value, where value is friendly tiles - enemy tiles.
        reward_maps: NDArray = post_step_value_maps - pre_step_value_maps
        # reward_maps: NDArray = post_step_value_maps

        if not done:
            done = sum(np.unique(self.grid) >= Tiles.AGENT) < 2  # end if there is only one kind of agent territory left

        n_obs = self._get_obs()
        info = self._get_info()

        if self.window:  # if pygame video system is initialized
            for event in pygame.event.get():  # finally check pending pygame events
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.window = None
                    done = True

                if event.type == pygame.KEYUP:
                    pass

        return n_obs, reward_maps, done, info

    def render(self, mode: str = 'human', fps: int = metadata['render_fps']) -> Union[NDArray, None]:
        if self.window is None and mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption('Competitive Game of Life')
        if self.clock is None and mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill(self.TILE_COLORS[Tiles.EMPTY])

        # first we draw in the tiles
        for i, axis0 in enumerate(self.grid):
            for j, tile in enumerate(axis0):
                if tile != Tiles.EMPTY:
                    pygame.draw.rect(
                        canvas,
                        self.TILE_COLORS[tile],
                        pygame.Rect(
                            (i * self.tile_height, j * self.tile_height),
                            self.tile_size,
                        ),
                    )

        if mode == 'human':
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined frame rate.
            # The following line will automatically add a delay to keep the frame rate stable.
            self.clock.tick(fps)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _num_tiles(self, grid) -> NDArray:
        unique_tiles, count_tiles = np.unique(grid, return_counts=True)
        num_tiles = np.zeros((self.n_tile_types,), dtype=np.uintc)
        num_tiles[unique_tiles] = count_tiles

        return num_tiles

    def _advantage(self, grid) -> NDArray:
        advantage = np.zeros((self.n_tile_types,), dtype=np.intc)
        num_tiles = self._num_tiles(grid)
        total_agent_tiles = num_tiles[Tiles.AGENT:].sum()
        for agent_id in range(Tiles.AGENT, self.n_tile_types):
            enemy_tiles = total_agent_tiles - num_tiles[agent_id]
            avg_enemy_tiles = enemy_tiles / (self.n_agents - 1)
            advantage[agent_id] = num_tiles[agent_id] - avg_enemy_tiles

        return advantage

    def run_game(self, max_steps: Union[int, None] = None, render_mode: Union[str, None] = 'human', fps: int = 2) -> Sequence[GameState]:
        self.max_timestep = max_steps if max_steps is not None else self.DEFAULT_MAX_TIMESTEP

        n_obs: Sequence[NDArray] = self.reset()
        done = False

        last_num_tiles = np.zeros((self.n_tile_types,), dtype=np.uintc)
        last_advantage = np.zeros((self.n_tile_types,), dtype=np.intc)
        cum_attacked = np.zeros(self.shape, dtype=np.ulonglong)

        game_history: MutableSequence[GameState] = []

        while not done:
            if self.timestep % 100 == 0:
                print(self.timestep)
            if render_mode:
                self.render(mode=render_mode, fps=fps)

            current_grid = np.copy(self.grid)
            n_action: MutableSequence[Union[NDArray, None]] = [None] * self.n_agents
            for rel_agent_id, agent in enumerate(self.agents):
                n_action[rel_agent_id] = agent.policy(
                    n_obs[rel_agent_id],
                    self.action_spaces[rel_agent_id],
                    self.observation_spaces[rel_agent_id]
                )

            normalized_actions: Sequence[Union[NDArray, None]] = self._normalize_actions(n_action)

            n_obs, n_reward, done, info = self.step(n_action)

            attacked_grid = np.concatenate(  # add tiles.empty and tiles.wall dimensions as having attacked nothing
                (np.zeros((*self.shape, Tiles.AGENT), dtype=np.ushort), self._get_attacked_grid(normalized_actions)),
                axis=-1,
            )
            total_attacked_grid = attacked_grid.sum(axis=2)
            cum_attacked += total_attacked_grid
            num_tiles = self._num_tiles(current_grid)
            advantage = self._advantage(current_grid)
            game_state = GameState(
                grid=current_grid,
                attacks=np.concatenate(  # add tiles.empty and tiles.wall dimensions with default attacks
                    (np.zeros((Tiles.AGENT, *self.shape, 2), dtype=np.ushort), np.array(normalized_actions)),
                    axis=0,
                ),
                attacked=attacked_grid,
                total_attacked=total_attacked_grid,
                cum_attacked=np.copy(cum_attacked),
                num_tiles=attacked_grid.sum(axis=(0, 1)),
                num_tiles_delta=num_tiles.astype(np.int_) - last_num_tiles,
                advantage=advantage,
                advantage_delta=advantage - last_advantage,
            )

            last_num_tiles = num_tiles
            last_advantage = advantage

            game_history.append(game_state)

        self.close()

        return game_history
