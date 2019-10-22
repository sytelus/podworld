import gym
import ray
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from podworld.envs import PodWorldEnv

import tensorwatch as tw
import time
start_time = time.time()
print('Started', start_time)
watcher = tw.Watcher(filename='ppo_agent.log')
logger = watcher.create_stream(name='reward')
watcher.make_notebook()

ray.init(num_gpus=1)
np.seterr(all='raise')

config = DEFAULT_CONFIG.copy()
config.update({
    "lambda": 0.95,
    "kl_coeff": 0.5,
    "clip_rewards": True,
    "clip_param": 0.1,
    "vf_clip_param": 10, #10
    "entropy_coeff": 0.01,

    "batch_mode": "truncate_episodes",
    "observation_filter": "NoFilter",
    "vf_share_layers": True,

    "train_batch_size": 50000,
    "sample_batch_size": 20,
    "sgd_minibatch_size": 500,
    "num_sgd_iter": 10,

    "num_workers": 1, # 32
    "num_envs_per_worker": 1, #5

    "num_gpus": 1,

    "model": {"dim":64}
    })

def env_creator(env_config):
    return PodWorldEnv(max_steps=10000, reward_factor=10000.0)  
register_env("podworld_env", env_creator)
agent = PPOTrainer(config=config, env="podworld_env")
agent_save_path = None

for i in range(50):
    stats = agent.train()
    # print(pretty_print(stats))
    if i % 5 == 0 and i > 0:
        path = agent.save()
        if agent_save_path is None:
            agent_save_path = path
            print('Saved agent at', agent_save_path)
    logger.write((i, stats['episode_reward_min']))
    print ('episode_reward_mean', stats['episode_reward_min'])
