from battle.envs import GridBattle
from battle.util import createN, create1v1, Tiles, GameHistoryJSONEncoder
from itertools import combinations
from collections import namedtuple
from tournament_agents import AGENTS
import numpy as np
import json
from PIL import Image
import os


def file_1v1_to_ndarray(path):
    # load map files
    img = Image.open(path)

    img_np = np.array(img)
    walls = img_np[..., 0] > 250
    start_1 = img_np[..., 1] > 250
    start_2 = img_np[..., 2] > 250

    board = np.zeros(img_np.shape[:-1], dtype=np.ushort)
    board[walls] = Tiles.WALL
    board[start_1] = Tiles.AGENT
    board[start_2] = Tiles.AGENT + 1

    return board


RoundFormat = namedtuple('RoundFormat', ('map', 'timesteps', 'name'))

map1v1s = [
    RoundFormat(file_1v1_to_ndarray('maps\\tennis.png'), 1000, 'tennis'),
    RoundFormat(file_1v1_to_ndarray('maps\\stripe.png'), 500, 'stripe'),
    RoundFormat(create1v1(5), 50, 'small'),
    RoundFormat(file_1v1_to_ndarray('maps\\cells.png'), 2000, 'cells'),
    RoundFormat(file_1v1_to_ndarray('maps\\court.png'), 1000, 'court'),
]

mapffa_1 = RoundFormat(createN(50, len(AGENTS)), 2000, 'large')

map1v1Customs = [
    RoundFormat(file_1v1_to_ndarray('maps\\circles.png'), 2000, 'circles'),
]

STANDARD_MAP_ROUNDS = 4
FFA_MAP_ROUNDS = 4
CUSTOM_MAP_ROUNDS = 2
EXPORT_ROOT = 'games'


scores = [0] * len(AGENTS)


def winner1v1(game_history):
    return int(np.argmax(game_history[-1].num_tiles[2:]))


def ffaEndPoints(game_history):
    return np.argsort(game_history[-1].num_tiles[2:])


for (i, agent1), (j, agent2) in combinations([(i, agent) for i, agent in enumerate(AGENTS)], 2):
    for map1v1 in map1v1s:
        for k in range(STANDARD_MAP_ROUNDS):
            print(f'running {k} iteration {map1v1.name} map between {i} and {j} for {map1v1.timesteps} timesteps')
            order = (agent1, agent2) if k % 2 == 0 else (agent2, agent1)
            env = GridBattle((agent1, agent2), map1v1.map)

            game_history = env.run_game(map1v1.timesteps, render_mode=None)
            winner = winner1v1(game_history)

            scores[(i, j)[winner]] += 1

            with open(os.path.join(EXPORT_ROOT, f'agent{i}_v_agent{j}_map_{map1v1.name}_iteration_{k}.json'), 'w') as history_file:
                history_file.write(json.dumps(game_history, cls=GameHistoryJSONEncoder))

    for map1v1Custom in map1v1Customs:
        for k in range(CUSTOM_MAP_ROUNDS):
            print(f'running {k} iteration {map1v1Custom.name} map between {i} and {j} for {map1v1Custom.timesteps} timesteps')
            order = (agent1, agent2) if k % 2 == 0 else (agent2, agent1)
            env = GridBattle((agent1, agent2), map1v1Custom.map)

            game_history = env.run_game(map1v1Custom.timesteps, render_mode=None)
            winner = winner1v1(game_history)

            scores[(i, j)[winner]] += 1

            with open(os.path.join(EXPORT_ROOT, f'agent{i}_v_agent{j}_map_{map1v1Custom.name}_iteration_{k}.json'), 'w') as history_file:
                history_file.write(json.dumps(game_history, cls=GameHistoryJSONEncoder))

for i in range(FFA_MAP_ROUNDS):
    print(f'running {i} iteration {mapffa_1.name} ffa map for {mapffa_1.timesteps} timesteps')
    env = GridBattle(AGENTS, mapffa_1.map)

    game_history = env.run_game(mapffa_1.timesteps, render_mode=None)
    points = ffaEndPoints(game_history)
    winners = points[len(AGENTS) // 2:]

    for winner in winners:
        scores[winner] += 1

    with open(os.path.join(EXPORT_ROOT, f'ffa_map_{mapffa_1.name}_iteration_{i}.json'), 'w') as history_file:
        history_file.write(json.dumps(game_history, cls=GameHistoryJSONEncoder))

print(scores)
