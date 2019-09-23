import gym
from gym import spaces

from ..render.renderer import Renderer 
from ..physics.world import Body
from ..physics.world import World
from ..physics.box_body import BoxBody
from ..physics.circle_body import CircleBody

import random
import math
import numpy as np
from typing import Iterator, Tuple

class BoxWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 60} #TODO: support 'ansi'

    def __init__(self, food_count=10, obs_count=30, xmax=2560, ymax=1440, seed=42,
        box_size=(40, 40), circle_radius=30, box_mass=1, circle_mass=1,
        bar_count=4, bar_size=(900,40), bar_mass=10,
        agent_radius=40, agent_mass=100, agent_obs_length=400, agent_ray_count=64)->None:

        np.random.seed(seed)
        random.seed(seed)

        self.food_count, self.obs_count, self.xmax, self.ymax = food_count, obs_count, xmax, ymax
        self.box_size, self.circle_radius, self.box_mass, self.circle_mass = box_size, circle_radius, box_mass, circle_mass
        self.bar_count, self.bar_size, self.bar_mass = bar_count, bar_size, bar_mass
        self.agent_radius, self.agent_mass, self.agent_obs_length = agent_radius, agent_mass, agent_obs_length
        self.agent_ray_count = agent_ray_count

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, agent_ray_count, 3), dtype=np.uint8)

        self._obs_local_pts = list(self._get_obs_local_pts())
        self._agent_mask = 0b1
        self.last_observation = None

        self.renderer = Renderer()

    def _get_obs_local_pts(self)->Iterator[Tuple[float, float]]:
        start = 0
        step = 2 * math.pi / self.observation_space.shape[1]
        for i in range(self.observation_space.shape[1]):
            angle = start + step * i
            yield math.cos(angle) * self.agent_obs_length, math.sin(angle) * self.agent_obs_length

    def _get_random_position(self, clearance)->tuple:
        return random.randint(clearance, self.xmax-clearance), random.randint(clearance, self.ymax-clearance)

    def step(self, action):
        self.world.step(1.0/self.renderer.human_mode_fps)
        observations = list(self.world.get_observations(self.agent, self._obs_local_pts, self._agent_filter))
        self.last_observation =  np.expand_dims(np.array(observations, dtype=np.uint8), axis=0)
        return self.last_observation, None, self.renderer.exited or False, None

    def _get_random_circle_radius(self)->float:
        return max(5.0, abs(random.normalvariate(self.circle_radius, self.circle_radius/2)))
    def _get_random_box_size(self)->float:
        return (max(5.0, abs(random.normalvariate(self.box_size[0], self.box_size[0]/2))),
            max(5.0, abs(random.normalvariate(self.box_size[1], self.box_size[1]/2))))
    def _get_random_bar_size(self)->float:
        return (max(10.0, abs(random.normalvariate(self.bar_size[0], self.bar_size[0]/4))),
            max(10.0, abs(random.normalvariate(self.bar_size[1], self.bar_size[1]/8))))

    def reset(self):
        self.obss, self.foods, self.bars = [], [], []
        self.agent = None
        self.world = World(xmax=self.xmax, ymax=self.ymax)
        self.world.create_boundry()

        for _ in range(self.obs_count):
            radius = self._get_random_circle_radius()
            obj = CircleBody(mass=self.circle_mass * radius * radius, 
                position=self._get_random_position(clearance=self.circle_radius),
                angle=random.random() * 2 * math.pi,
                radius=radius, rgba_color=(162, 110, 180,255))
            self.world.add(obj)
            self.foods.append(obj)            
            obj._apply_gaussian_impulse((self.circle_radius, self.circle_radius), obj.mass, obj.mass*100)
        for _ in range(self.food_count):
            box_size = self._get_random_box_size()
            obj = BoxBody(mass=self.box_mass * box_size[0] * box_size[1], 
                position=self._get_random_position(clearance=max(self.box_size)),
                angle=random.random() * 2 * math.pi,
                size=box_size, corner_radius=1.0, rgba_color= (74, 198, 73, 255))
            self.world.add(obj)
            self.obss.append(obj)
            obj._apply_gaussian_impulse(self.box_size, obj.mass, obj.mass*100)
        for i in range(self.bar_count):
            bar_size = self._get_random_bar_size()
            obj = BoxBody(mass=self.bar_mass * bar_size[0] * bar_size[1], 
                position=(self.xmax/(self.bar_count+1)*(i+1), random.randint(0,self.ymax)),
                angle=random.random() * 2 * math.pi,
                size=bar_size, corner_radius=1.0, rgba_color= (232, 184, 164, 255))
            self.world.add(obj)
            self.bars.append(obj)
            #obj._apply_gaussian_impulse(self.box_size, 0, obj.mass*100)

        # add agent
        self._agent_filter = World.get_filter(self._agent_mask)        
        self.agent = CircleBody(mass=self.agent_mass, 
            position=self._get_random_position(clearance=self.agent_radius),
            angle=random.random() * 2 * math.pi,
            radius=self.agent_radius, rgba_color=(224, 23, 33,255), category_mask=self._agent_mask)
        self.world.add(self.agent)

    def render(self, mode='human'):
        self.renderer.render(self.world, self.last_observation, mode=mode)

    def close(self):
        self.obss, self.foods = [], []
        self.world = None
