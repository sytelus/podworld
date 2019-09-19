import gym
from gym import error, spaces, utils
from gym.utils import seeding

from ..physics.world import World
from ..physics.box_body import BoxBody
from ..physics.circle_body import CircleBody

import random
import 

class BoxWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']} #TODO: support 'ansi'

    def __init__(self, food_count=10, obs_count=10, xmax=1280, ymax=720, 
        box_size=(20, 20), circle_radius=30, box_mass=10, circle_mass=20)->None:

        self.food_count, self.obs_count, self.xmax, self.ymax = food_count, obs_count, xmax, ymax
        self.box_size, self.circle_radius, self.box_mass, self.circle_mass = box_size, circle_radius, box_mass, circle_mass

        self.world = World(xmax=xmax, ymax=ymax)
        self.world.create_boundry()

        self.obss, self.foods = [], []

        for _ in range(self.obs_count):
            obs = BoxBody(mass=box_mass, position=self._get_random_position(clearance=max(self.box_size)),
                corner_radius=1.0)
            self.world.add(obs)
            self.obss.append(obs)
            obs.apply_impulse()
        for _ in range(self.food_count):
            food = CircleBody(mass=circle_mass, position=self._get_random_position(clearance=self.circle_radius),
                radius=circle_radius)
            self.world.add(food)
            self.foods.append(food)            

    def _get_random_position(self, clearance)->tuple:
        return random.randint(clearance, self.xmax-clearance), random.randint(clearance, self.ymax-clearance)

    def step(self, action):
    ...
    def reset(self):
    ...
    def render(self, mode='human'):
    ...
    def close(self):
    ...