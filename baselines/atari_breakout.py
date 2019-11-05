# this script is for experimenting with Atari Breakout and RLLib

import ray
import numpy as np
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.logger import pretty_print
import yaml

from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.visionnet_v2 import VisionNetwork

ModelCatalog.register_custom_model("my_model", VisionNetwork)

config = None
with open('../rl-experiments/atari-dqn/dueling-ddqn.yaml') as f:
    experiments = yaml.safe_load(f)
    exp_name = next(iter(experiments)) # first key
    config = experiments[exp_name]['config']
    config['num_gpus']=1
    config['model'] = {
            "custom_model": "my_model",
            "custom_options": {},  # extra options to pass to your model
        }

ray.init(num_gpus=1, num_cpus=1)

agent = DQNTrainer(config=config, env="BreakoutNoFrameskip-v4")
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
