import gym
from gym import spaces
import numpy as np
from typing import Tuple, List
from numpy.typing import NDArray
from collections.abc import Sequence, Iterable
from collections import Counter
from typeguard import check_argument_types
from ..util import *
import pygame


class GridBattleHuman(gym.Env):
    """
    This is the grid battle environment for the MLDS 2022-2023 long competition.

    This is a multi-agents environment, meaning the returned observation is actually a tuple of agents observations.
    From here on, when the documentation refers to an 'observation' or 'action', these refer to each agents's
    observations and actions. For example, if the environment returns an observation = (obs0, obs1, obs2), then the
    observation space describes the space of values obs0 could be.

    Let A be the set of agents, and let N be the size of A.
    Let K be the set of cells that an agents has, and let D_k be the size of K.

    The action is a numpy array of floats with shape D_k * H * W (PyTorch format 😎)
    1. D_k: Cells, the number of cells a particular agents has on the grid (this number can vary as the game progresses)
    3. H: Height, the height of the vision window
    4. W: Width, the width of the vision window

    The environment's defined action_space is H * W, meaning that you would need to sample from it D times to get a
    random action for each cell an agents has. To produce random actions for all agents, you would need to sample N * D_k
    times.

    The action should describe a probability distribution for what cells to attack. If you want, you could also set
    use_logits=True in the constructor to pass in logits instead of a distribution. This allows you to set the
    `env.temperature` (assuming `env` is your environment object) to control how the actions are sampled.

    The observation is a numpy array of floats with shape D_k * C * H * W
    1. D_k: Cells, the number of cells a particular agents has on the grid (this number can vary as the game progresses)
    2. C: Channel, the number of agents in the game + 2 (the extra two channels are for empty and wall tiles)
    3. H: Height, the height of the vision window
    4. W: Width, the width of the vision window

    The channel axis is a one hot encoded representation of the tile state. E.g. for a given cell (D_k=1) at a
    particular tile (H=1, W=1), we get a 1 * C * 1 * 1 which can be squeezed into a C-dimensional vector that encodes
    the state of that particular tile for that particular cell. So if the vector was [0, 0, 1, 0, 0], it would tell us
    that there is currently a friendly cell at that position, but if it was [0, 0, 0, 1, 0], it would mean there is an
    enemy cell. Look into the Tiles enum within the battle.util.types file to see what the different one-hot encodings
    represent.

    The environment's defined observation_space is C * H * W, meaning that you would need to sample from it D_k times to
    get a random observation for each cell an agents has. To produce random observations for all agents, you would need
    to sample \sum^N_{k=1}D_k times.
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 2,
    }

    DEFAULT_MAX_TIMESTEP = 200
    TILE_COLORS = (  # colors to assign to each tile type when displaying
        (255, 255, 255),
        (50, 50, 50),
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 127, 0),
        (127, 255, 0),
        (0, 127, 255),
        (255, 0, 127),
    )
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

    action_space: spaces.Box
    observation_space: spaces.MultiBinary


    def __init__(self,
                 initial_map: NDArray[int] | Sequence[int, int] | int = 8,
                 vision_window: Sequence[int, int] | int = (3, 3),
                 attack_window: Sequence[int, int] | int | None = None,
                 use_logits: bool = False,
                 temperature: float = 1,
                 window_height: int = 560,
                 ) -> None:
        """
        Create the grid battle environment. The vision window determines the H and W axes of the observation space,
        which will not change in the future.

        :param initial_map: An initial map (H * W enum-encoded numpy array), a height/width tuple, or a side length int
        :type initial_map: NDArray[int] | Sequence[int, int] | int
        :param vision_window: A height/width tuple or side length int that sets cells' H * W vision window dimensions
        :type vision_window: Sequence[int, int] | int
        :param vision_window: An optional height/width tuple or side length int for cells' attack window dimensions
        :type vision_window: Sequence[int, int] | int | None
        :param use_logits: Whether the action space is a distribution or logits.
        :type use_logits: bool
        :param temperature: The temperature to sample at if the action space is in logits.
        :type temperature: float
        :param window_height: The height of the rendering window in pixels
        :type window_height: int
        """
        super(GridBattleHuman, self).__init__()
        if not isinstance(initial_map, np.ndarray):
            initial_map: NDArray[int] = create1v1(initial_map)
        if isinstance(vision_window, int):
            vision_window: Tuple[int, int] = (vision_window, vision_window)
        if isinstance(attack_window, int):
            attack_window: Tuple[int, int] = (attack_window, attack_window)

        self.vision_window = vision_window  # H * W window of what the bot can see
        if attack_window is None:
            self.attack_window = self.vision_window
        self.initial_map = np.copy(initial_map)  # store a copy of this that we can reset to in the future
        self.shape: Tuple = initial_map.shape  # grid shape
        self.max_timestep = self.DEFAULT_MAX_TIMESTEP

        self.tile_width = window_height // self.shape[0]
        self.tile_size = (self.tile_width, self.tile_width)  # size of each tile in pixels
        height = self.shape[0] * self.tile_width
        width = self.tile_width * self.shape[1]
        self.window_size = (height, width)
        self._use_logits = use_logits
        self.temperature = temperature
        self.timestep = 0
        self.seed = None

        self.grid: NDArray[int] = np.empty(self.shape, dtype=np.byte)
        self.n_agents = np.amax(initial_map) + 1 - Tiles.AGENT  # number of agents determined by initial map tile values

        if self._use_logits:
            self.action_space: spaces.Box = spaces.Box(low=-np.inf, high=np.inf, shape=self.attack_window)
        else:
            self.action_space: spaces.Box = spaces.Box(low=0.0, high=1.0, shape=self.attack_window)

        # the shape of a given cell's observation
        self._obs_shape = (self.n_agents + Tiles.AGENT, *self.vision_window)

        self.observation_space: spaces.MultiBinary = spaces.MultiBinary(self._obs_shape)


        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct frame rate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self) -> Sequence[NDArray]:
        """
        Gets a full observation. This should only need ot be called once per step, it supplies an observation for each
        agents. Each observation (np array) follows the observation space standard, meaning the first axis is of
        variable size.

        :return: A list of agents observations that in the observation space. The return value itself is NOT in the
        observation space.
        :rtype: List[NDArray]
        """
        agent_observations = []
        global_agent_mask: NDArray = self.grid == Tiles.AGENT  # mask of cells belonging to agents Tiles.AGENT
        for agent_id in map(lambda x: x + Tiles.AGENT, range(self.n_agents)):
            agent_mask: NDArray = self.grid == agent_id  # mask of cells belonging to agents agent_id
            cell_observations: List[NDArray] = []  # axis 0 of agents observation space, each cell's observation

            perspective_grid = np.copy(self.grid)  # a copy with our agent_id swapped with agents of id Cells.AGENT
            perspective_grid[global_agent_mask] = agent_id
            perspective_grid[agent_mask] = Tiles.AGENT

            for i, axis0 in enumerate(agent_mask):  # essentially a dataloader for each cell observation
                for j, tile in enumerate(axis0):
                    if not tile:  # yea this is a bit inefficient, not that much tho cuz not many agents. i think.
                        continue

                    view: NDArray = np.full(self.vision_window, Tiles.WALL)  # create cell view array
                    # the offset of where the cell is relative to (0, 0) in the view
                    view_offset = tuple((view_size_axis - 1) // 2 for view_size_axis in self.vision_window)
                    view_pos = (i - view_offset[0], j - view_offset[1])  # the top left corner of the bot's view

                    grid_intersect = (  # area of intersection between global grid and bot view, global perspective
                        slice(max(0, view_pos[0]), min(perspective_grid.shape[0], view_pos[0] + self.vision_window[0])),
                        slice(max(0, view_pos[1]), min(perspective_grid.shape[1], view_pos[1] + self.vision_window[1])),
                    )
                    view_intersect = (  # area of intersection between global grid and bot view, local perspective
                        slice(max(0, -view_pos[0]),
                              self.vision_window[0] +
                              min(0, perspective_grid.shape[0] - (view_pos[0] + self.vision_window[0]))
                              ),
                        slice(max(0, -view_pos[1]),
                              self.vision_window[0] +
                              min(0, perspective_grid.shape[1] - (view_pos[1] + self.vision_window[1]))
                              ),
                    )

                    view[view_intersect[0], view_intersect[1]] = \
                        perspective_grid[grid_intersect[0], grid_intersect[1]]
                    # now we should have a correct cell observation that is enum-encoded, so we gotta one-hot encode now
                    # https://stackoverflow.com/questions/63839024/one-hot-encode-numpy-array-with-2-dims
                    cell_observation = np.zeros(self._obs_shape, dtype=bool)
                    cell_observation[(view, *np.indices(self.vision_window, sparse=True))] = True

                    cell_observations.append(cell_observation)  # finished one-hot encoding view to C axis, add data

            agent_observations.append(np.ascontiguousarray(cell_observations, dtype=bool))  # make obs into np array

        return agent_observations

    def _get_info(self) -> dict:
        return {
            'grid': self.grid
        }

    # checks if a 2d point is a valid point to attack
    def _valid_cell(self, pos: Iterable | Sequence[int, int] | NDArray[int]) -> bool:
        for axis in range(2):
            if not 0 <= pos[axis] < self.grid.shape[axis]:  # if out of bounds in some axis
                return False

        return self.grid[pos] != Tiles.WALL

    def reset(self,
              *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None,
              ) -> Tuple[Sequence[NDArray], dict] | Sequence[NDArray]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.grid = np.copy(self.initial_map)  # reset grid
        self.seed = seed

        self.max_timestep = options['max_timestep']if isinstance(options, dict) and 'max_timestep' in options else \
            self.DEFAULT_MAX_TIMESTEP

        n_observation = self._get_obs()
        info = self._get_info()
        return (n_observation, info) if return_info else n_observation

    def step(self, action: Sequence[NDArray]) -> Tuple[Sequence[NDArray], Sequence[int], bool, dict]:
        """
        The rewards are given on the basis of agents meaning each agents gets a single scalar reward. The reward is
        calculated as the change in map control, where map control is the difference in friendly cells and enemy cells.
        e.g. if in state S there are 3 friendly cells, 4 empty tiles, and 2 enemy cells, the map control
        is 3 - 2 = 1. Then, if the next state S' contains 4 friendly cells, 1 empty tile, and 4 enemy tiles, the map
        control is 4 - 4 = 0. Then the difference in map control within the cell's view is 0 - 1 = -1, because although
        the cell's view gained one friendly cell, it also gained two enemy cells, resulting in a net change of -1.

        :param action: A list of actions for each agents to take, where each agents's actions is in the env action space.
        :type action: Sequence[NDArray]
        :return: A list of observations, a list of rewards, whether the environment is done or not, and information.
        :rtype: Tuple[Sequence[NDArray], Sequence[NDArray], bool, dict]
        """
        assert check_argument_types(), \
            'action must be a sequence of agents actions'

        # assert there is an action for each agents
        assert len(action) == self.n_agents, 'there must be an action for every agents'

        # assert each agents's action matches the shape we expect
        for rel_agent_id, agent_action in enumerate(action):
            assert agent_action.shape[1:] == self.attack_window, \
                f'action must match action space, agents {rel_agent_id + Tiles.AGENT}\'s action is not the correct shape'

        # actions where the H * W axes are flattened to one axis, making it easier to sample.
        flattened_actions = [agent_action.reshape((agent_action.shape[0], -1)) for agent_action in action]

        # list of each agents's cell rewards, which are stored as np arrays
        n_reward = []

        # grid that shows how much each tile is being attacked by each agents
        attacked_grid = np.zeros((self.n_agents, *self.shape), dtype=np.ushort)

        # first, sample each cell's attack probability distribution
        # iterate in the same way that the observations are iterated over, so we have the same indices
        cell_ids = [0] * self.n_agents  # running count of the cell indexes for each agents's action
        attack_offset = tuple((attack_size_axis - 1) // 2 for attack_size_axis in self.attack_window)
        for i, axis0 in enumerate(self.grid):
            for j, tile in enumerate(axis0):
                if tile < Tiles.AGENT:
                    continue

                rel_agent_id = tile - Tiles.AGENT
                cell_id = cell_ids[rel_agent_id]
                # sample from 2d distribution https://stackoverflow.com/questions/56017163
                distribution = flattened_actions[rel_agent_id][cell_id]

                # Then, sample an index from the 1D array with the
                # probability distribution from the original array
                sample_index = np.random.choice(a=distribution.size, p=distribution)

                # Take this index and adjust it, so it matches the original array
                index_2d = np.unravel_index(sample_index, self.attack_window)

                global_index = tuple(global_idx + relative_idx - offset
                                     for relative_idx, offset, global_idx in zip(index_2d, attack_offset, (i, j)))

                if self._valid_cell(global_index):
                    attacked_grid[(rel_agent_id, *global_index)] += 1  # indicate the tile is being attacked

                cell_ids[rel_agent_id] += 1

        # by now the attacked_grid should be populated with the tiles that are attacked by each agents

        # a grid that says the max times each tile was attacked by any one team
        max_attacked_grid = attacked_grid.max(axis=0, initial=0)
        # https://stackoverflow.com/questions/42071597/numpy-argmax-random-tie-breaking
        converted_grid = np.argmax(np.random.random(attacked_grid.shape) * (attacked_grid == max_attacked_grid),
                                   axis=0) + Tiles.AGENT  # transform from relative agents ids to actual agents ids

        # tied_grid = np.sum(attacked_grid == max_attacked_grid, axis=0) > 1  # implement empty tie-breaking

        unconverted_mask = max_attacked_grid == 0  # mask of tiles that were not attacked which should not be converted
        converted_mask = max_attacked_grid > 0  # only convert the tiles that were attacked at least once
        converted_grid[unconverted_mask] = 0

        new_grid = np.copy(self.grid)
        new_grid[converted_mask] = 0
        new_grid += converted_grid
        # new_grid[tied_grid] = Tiles.EMPTY

        old_counts = Counter(self.grid.reshape([-1]))
        old_non_cell = sum([old_counts[i] for i in range(Tiles.AGENT)])  # number of non-cell (wall or empty) tiles
        new_counts = Counter(new_grid.reshape([-1]))
        new_non_cell = sum([new_counts[i] for i in range(Tiles.AGENT)])
        total = self.shape[0] * self.shape[1]  # for a given agents, the # of enemy cells is total - non-cell - friendly

        for agent_id in map(lambda x: x + Tiles.AGENT, range(self.n_agents)):
            num_old_friendly = old_counts[agent_id]
            num_old_enemy = total - old_non_cell - num_old_friendly

            old_map_control = num_old_friendly - num_old_enemy

            num_new_friendly = new_counts[agent_id]
            num_new_enemy = total - new_non_cell - num_new_friendly
            new_map_control = num_new_enemy - num_new_enemy

            n_reward.append(new_map_control - old_map_control)

        self.grid = new_grid

        self.timestep += 1
        done = self.timestep >= self.max_timestep  # end if time has run out
        if not done:
            done = sum(np.unique(self.grid) >= Tiles.AGENT) < 2  # end if there is only one kind of agents territory left

        n_observation = self._get_obs()
        info = self._get_info()

        if self.window:  # if pygame video system is initialized
            for event in pygame.event.get():  # finally check pending pygame events
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.window = None
                    done = True

                if event.type == pygame.KEYUP:
                    pass

        return n_observation, n_reward, done, info

    def render(self, mode: str = 'human') -> NDArray | None:
        if self.window is None and mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
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
                            (i * self.tile_width, j * self.tile_width),
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
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
