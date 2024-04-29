import numpy as np
import math
from typing import Tuple, Union
from numpy.typing import NDArray
from .types import Tiles
from functools import partial

# Util functions for creating maps


def createEmpty(shape: Union[Tuple[int, int], int]) -> NDArray[int]:
    """
    Creates an empty map of specified shape

    :param shape: Either the height/width of the grid or the side length
    :type shape: Union[Tuple[int, int], int]
    :return: An empty map of the given size
    :rtype: NDArray
    """
    if isinstance(shape, int):
        shape = (shape, shape)

    return np.expand_dims(np.full(shape, Tiles.EMPTY, dtype=np.ushort), axis=0)


def createN(shape: Union[Tuple[int, int], int], num_agents: int) -> NDArray[int]:
    """
    Creates a map of specified shape that is empty except for starting positions for N agents

    :param shape: Either the height/width of the grid or the side length
    :type shape: Union[Tuple[int, int], int]
    :param num_agents: The number of agents to start with
    :type num_agents: int
    :return: A n-agent battle royale map of the given size
    :rtype: NDArray
    """
    if isinstance(shape, int):
        shape = (shape, shape)

    new_map = createEmpty(shape)

    agents_per_side = 1 + math.isqrt(num_agents - 1)
    max_possible_agents = agents_per_side ** 2

    for i in range(num_agents):
        if i % 2 == 0:
            i_pos = i // 2
        else:
            i_pos = max_possible_agents - ((i + 1) // 2)
        agent_id = i + Tiles.AGENT
        agent_axis_0 = ((i_pos // agents_per_side) * shape[0]) // agents_per_side
        agent_axis_1 = ((i_pos % agents_per_side) * shape[1]) // agents_per_side

        new_map[0, agent_axis_0, agent_axis_1] = agent_id


    return new_map


# Creates a map of specified shape that is empty except for two cells to 1v1
create1v1 = partial(createN, num_agents=2)

def slideMap(map: NDArray[int], delta: NDArray[int]) -> NDArray[int]:
    """
    Slides a map each frame in a given direction defined by a delta vector/

    :param map: The map to slide
    :type map: NDArray
    :param direction: The direction to slide in
    :type direction: NDArray
    :return: A map that slides the given map in the given direction
    :rtype: NDArray
    """
    
    initial_map = map
    maps = []

    while (len(maps) <= 0) or (maps[-1] != initial_map).any():
        maps.append(np.roll(maps[-1] if len(maps) >= 1 else initial_map, delta, axis=(0, 1)))
    
    evolving_map = np.stack(maps, axis=0)
    return np.roll(evolving_map, 1, axis=0)  # shifts the map back one frame so that the first frame is the initial map


def slideMapWalls(map: NDArray[int], delta: NDArray[int]) -> NDArray[int]:
    """
    Slides only the walls on a map each frame in a given direction defined by a delta vector.

    :param map: The map to slide
    :type map: NDArray
    :param direction: The direction to slide in
    :type direction: NDArray
    :return: A map that slides the given map in the given direction
    :rtype: NDArray
    """

    wall_map = map == Tiles.WALL
    sliding_map = slideMap(wall_map, delta).astype(np.ushort)

    sliding_map[0] = map

    return sliding_map

sample_map_1 = np.array([
    [Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,    Tiles.WALL,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY],
    [Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,    Tiles.WALL,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY],
    [Tiles.EMPTY,   Tiles.EMPTY,   Tiles.AGENT,   Tiles.EMPTY,    Tiles.WALL,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY],
    [Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,    Tiles.WALL,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY],
    [Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,    Tiles.WALL,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY],
    [Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,    Tiles.WALL,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY],
    [Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,    Tiles.WALL,   Tiles.EMPTY, Tiles.AGENT+1,   Tiles.EMPTY,   Tiles.EMPTY],
    [Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,    Tiles.WALL,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY],
    [Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,    Tiles.WALL,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY,   Tiles.EMPTY],
])

sample_map_1_sliding = slideMapWalls(sample_map_1, (0, 1))
