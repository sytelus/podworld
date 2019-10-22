from base_agent import BaseAgent 

class RandomAgent(BaseAgent):
    def reset(self, env):
        self.action_space = env.action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

