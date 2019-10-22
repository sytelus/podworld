import gym
import ray
import numpy as np
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from podworld.envs import PodWorldEnv

import tensorwatch as tw
import time
start_time = time.time()
print('Started', start_time)
watcher = tw.Watcher(filename='dqn_agent.log')
logger = watcher.create_stream(name='reward')
watcher.make_notebook()

ray.init(num_gpus=1)
np.seterr(all='raise')

config = DEFAULT_CONFIG.copy()
config.update({
    "gamma": 0.99,
    "lr": 0.0001,
    "learning_starts": 10000,
    "buffer_size": 50000,
    "sample_batch_size": 4,
    "train_batch_size": 320,
    "schedule_max_timesteps": 2000000,
    "exploration_final_eps": 0.01,
    "exploration_fraction": 0.1,

    "model": {"dim":64}
    })

def env_creator(env_config):
    return PodWorldEnv(max_steps=100, reward_factor=1.0)  
register_env("podworld_env", env_creator)
agent = DQNTrainer(config=config, env="podworld_env")
agent_save_path = None

for i in range(50):
    stats = agent.train()
    # print(pretty_print(stats))
    if i % 10 == 0 and i > 0:
        path = agent.save()
        if agent_save_path is None:
            agent_save_path = path
            print('Saved agent at', agent_save_path)
    logger.write((i, stats['episode_reward_min']))
    print ('episode_reward_mean', stats['episode_reward_min'])
