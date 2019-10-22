from podworld.envs import PodWorldEnv
from rl_loop import run_episode

from base_agent import BaseAgent 

class RandomAgent(BaseAgent):
    def reset(self, env):
        self.action_space = env.action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

env = PodWorldEnv(max_steps=10000)
agent = RandomAgent()

run_episode(env, agent)

