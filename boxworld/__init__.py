# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from gym.envs.registration import register

register(
    id='boxworld-v0',
    entry_point='boxworld.envs:BoxWorldEnv',
)
