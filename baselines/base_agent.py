import abc

class BaseAgent(metaclass=abc.ABCMeta):
    def reset(env):
        pass
        
    @abc.abstractmethod
    def act(observation, reward, done):
        pass