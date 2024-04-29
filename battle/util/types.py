from enum import IntEnum
from numpy.typing import NDArray
from typing import NamedTuple


# in the main environment, this is stored with an np.byte, so it can only go up to 127 (126 agents max)
class Tiles(IntEnum):
    EMPTY = 0  # unclaimed, no bot or wall on this tile
    WALL = 1  # tile with a wall on it
    AGENT = 2  # local agents (each agents sees its own territory in this layer)
    # 3, 4, 5... are the ids of the other agents


# a single given state in the game history, a sequence of these is a full game history
class GameState(NamedTuple):
    grid: NDArray  # (H, W) with int enum encoded tiles
    attacks: NDArray  # (D, H, W, 2) for D types of tiles (agents + 2) showing the relative coordinates of their attacks
    attacked: NDArray  # (D, H, W) of how many times each tile was attacked by each agent in this game state
    total_attacked: NDArray  # (H, W) of how many total times each tile was attacked in this game state
    cum_attacked: NDArray  # (H, W) of how many total times each tile was attacked, cumulatively
    num_tiles: NDArray  # (D) showing how many of each type of tile there are
    num_tiles_delta: NDArray  # (D) num_tiles - last_state_num_tiles
    advantage: NDArray  # (D) num_tiles - avg(num_enemy_tiles)
    advantage_delta: NDArray  # (D) advantage - last_state_advantage
