from roberto_agent import priority_agent, rand_enemy_agent
from battle import AgentFunction

@AgentFunction
def random_agent(obs, action_space, obs_space):
    return action_space.sample()

roberto1 = priority_agent()
roberto2 = rand_enemy_agent()

henry1 = random_agent()
henry2 = random_agent()

# AGENTS = (henry1, roberto1, henry2, roberto2)
AGENTS = (roberto1, roberto1)
