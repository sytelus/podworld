from baselines.random_agent import RandomAgent
from podworld.envs import PodWorldEnv
from random_agent import RandomAgent
from custom_trainer import train

env = PodWorldEnv(max_steps=1000)
agent = RandomAgent()

train(env, agent)

