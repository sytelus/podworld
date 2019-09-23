import gym
from gym import error, spaces, utils
from gym.utils import seeding

from ..render.renderer import Renderer 
from ..physics.world import Body
from ..physics.world import World
from ..physics.box_body import BoxBody
from ..physics.circle_body import CircleBody

import random

class BoxWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']} #TODO: support 'ansi'

    def __init__(self, food_count=10, obs_count=10, xmax=1280, ymax=720, 
        box_size=(20, 20), circle_radius=30, box_mass=10, circle_mass=20)->None:

        self.food_count, self.obs_count, self.xmax, self.ymax = food_count, obs_count, xmax, ymax
        self.box_size, self.circle_radius, self.box_mass, self.circle_mass = box_size, circle_radius, box_mass, circle_mass
        self.renderer = Renderer()

    def _get_random_position(self, clearance)->tuple:
        return random.randint(clearance, self.xmax-clearance), random.randint(clearance, self.ymax-clearance)

    def step(self, action):
        self.world.step()
        return None, None, self.renderer.exited or False, None

    def reset(self):
        self.obss, self.foods = [], []
        self.world = World(xmax=self.xmax, ymax=self.ymax)
        self.world.create_boundry()

        for _ in range(self.obs_count):
            obs = BoxBody(mass=self.box_mass, position=self._get_random_position(clearance=max(self.box_size)),
                size=self.box_size, corner_radius=1.0)
            self.world.add(obs)
            self.obss.append(obs)
            obs._apply_gaussian_impulse(self.box_size, 10.0, 1000.0)
        for _ in range(self.food_count):
            food = CircleBody(mass=self.circle_mass, position=self._get_random_position(clearance=self.circle_radius),
                radius=self.circle_radius)
            self.world.add(food)
            self.foods.append(food)            
            food._apply_gaussian_impulse((self.circle_radius, self.circle_radius), 10.0, 1000.0)

    def render(self, mode='human'):
        self.renderer.render(self.world, mode)

    def close(self):
        self.obss, self.foods = [], []
        self.world = None
