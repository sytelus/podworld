# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from gym.envs.registration import register

register(
    id='podworld-v0',
    entry_point='podworld.envs:PodWorldEnv',
)
