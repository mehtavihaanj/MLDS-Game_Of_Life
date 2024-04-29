from numpy.typing import NDArray
from battle.envs import GridBattle
from battle.util import createN, create1v1, Tiles, GameHistoryJSONEncoder
from battle.agents import Agent, AgentFunction
import json
from roberto_agent import priority_agent, rand_enemy_agent

initial_map = createN(30, 2)


@AgentFunction
def myAgent(obs, action_space, obs_space) -> NDArray:
    return action_space.sample()


agent1 = myAgent()
agent2 = priority_agent()
agent3 = rand_enemy_agent()

env = GridBattle((agent2, agent3), initial_map)

game_history = env.run_game(500, render_mode=None)

with open('game.json', 'w') as history_file:
    history_file.write(json.dumps(game_history, cls=GameHistoryJSONEncoder))
