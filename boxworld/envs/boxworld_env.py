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

    OBJ_TYPE_ROCK = 1
    OBJ_TYPE_FOOD = 2
    OBJ_TYPE_BAR = 3
    OBJ_TYPE_WALL = 4
    OBJ_TYPE_AGENT = 0

    EATEN_FOOD_COLOR = (194, 199, 190, 255)
    AVAIL_FOOD_COLOR = (74, 198, 73, 255)


    def __init__(self, food_count=10, obs_count=30, xmax=2560, ymax=1440, seed=42,
        box_size=(40.0, 40.0), circle_radius=30.0, box_mass=1.0, circle_mass=1.0,
        bar_count=4, bar_size=(900.0,40.0), bar_mass=10.0,
        agent_radius=40.0, agent_mass=100.0, agent_obs_length=400.0, agent_ray_count=64,
        action_strength=50000.0, friction=0.0, elasticity=1.0)->None:

        self.seed(seed)

        self.obss, self.foods, self.bars, self.agent, self.world = \
            None, None, None, None, None
        self.eaten_foods = None
        self.step_reward, self.episod_reward, self.step_count = None, None, None
        self.initial_total_momentum, self.last_total_momentum = None, None
        
        self.food_count, self.obs_count, self.xmax, self.ymax = food_count, obs_count, xmax, ymax
        self.box_size, self.circle_radius, self.box_mass, self.circle_mass = box_size, circle_radius, box_mass, circle_mass
        self.bar_count, self.bar_size, self.bar_mass = bar_count, bar_size, bar_mass
        self.agent_radius, self.agent_mass, self.agent_obs_length = agent_radius, agent_mass, agent_obs_length
        self.agent_ray_count, self.action_strength = agent_ray_count, action_strength
        self.friction, self.elasticity = friction, elasticity

        self.action_space = spaces.Discrete(16+1) # 16 directions clockwise, action 0 is no op
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, agent_ray_count, 3), dtype=np.uint8)

        self._obs_local_pts = list(self._get_pts_on_circle(self.observation_space.shape[1], self.agent_obs_length))
        self._action_pts = list(self._get_pts_on_circle(self.action_space.n-1, self.action_strength))
        self._agent_mask = 0b1
        self.last_observation = None

        self.renderer = Renderer()

    def reset(self):
        self.obss, self.foods, self.bars = [], [], []
        self.eaten_foods = {}
        self.agent = None
        self.step_reward, self.episod_reward, self.step_count = 0.0, 0.0, 0
        if self.world:
            self.world.end()
        self.world = World(xmax=self.xmax, ymax=self.ymax)
        self.world.create_boundry(collision_type=BoxWorldEnv.OBJ_TYPE_WALL)
        self.world.set_collision_callback(BoxWorldEnv.OBJ_TYPE_AGENT, BoxWorldEnv.OBJ_TYPE_FOOD, 
            self._on_agent_food_collision)
        self.world.set_collision_callback(BoxWorldEnv.OBJ_TYPE_AGENT, BoxWorldEnv.OBJ_TYPE_ROCK, 
            self._on_agent_rock_collision)
        self.initial_total_momentum, self.last_total_momentum = None, None

        for _ in range(self.obs_count):
            radius = self._get_random_circle_radius()
            obj = CircleBody(mass=self.circle_mass * radius * radius, 
                position=self._get_random_position(clearance=10),
                angle=random.random() * 2 * math.pi, collision_type=BoxWorldEnv.OBJ_TYPE_ROCK,
                radius=radius, rgba_color=(162, 110, 180,255), 
                friction=self.friction, elasticity=self.elasticity)
            self.world.add(obj)
            self.obss.append(obj)            
            obj._apply_gaussian_impulse((radius, radius), obj.mass, obj.mass*100)
        for _ in range(self.food_count):
            box_size = self._get_random_box_size()
            mass = self.box_mass * box_size[0] * box_size[1]
            obj = BoxBody(mass=mass, 
                position=self._get_random_position(clearance=10),
                angle=random.random() * 2 * math.pi, collision_type=BoxWorldEnv.OBJ_TYPE_FOOD,
                size=box_size, corner_radius=1.0, rgba_color=BoxWorldEnv.AVAIL_FOOD_COLOR, 
                friction=self.friction, elasticity=self.elasticity)
            self.world.add(obj)
            self.foods.append(obj)
            obj._apply_gaussian_impulse(box_size, obj.mass, obj.mass*100)
        for i in range(self.bar_count):
            bar_size = self._get_random_bar_size()
            obj = BoxBody(mass=self.bar_mass * bar_size[0] * bar_size[1], 
                position=(self.xmax/(self.bar_count+1)*(i+1), random.randint(0,self.ymax)),
                angle=random.random() * 2 * math.pi, collision_type=BoxWorldEnv.OBJ_TYPE_BAR,
                size=bar_size, corner_radius=1.0, rgba_color= (232, 184, 164, 255), 
                friction=self.friction, elasticity=self.elasticity)
            self.world.add(obj)
            self.bars.append(obj)

        # add agent
        self._agent_filter = World.get_filter(self._agent_mask)        
        self.agent = CircleBody(mass=self.agent_mass, 
            position=self._get_random_position(clearance=self.agent_radius),
            angle=random.random() * 2 * math.pi, collision_type=BoxWorldEnv.OBJ_TYPE_AGENT,
            radius=self.agent_radius, rgba_color=(224, 23, 33,255), category_mask=self._agent_mask, 
            friction=self.friction, elasticity=self.elasticity)
        self.world.add(self.agent)

    # overriden method
    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def step(self, action):
        # perform action
        if action > 0:
            self.agent.apply_impulse(self._action_pts[action-1], (0,0))

        self.world.step(1.0/self.renderer.human_mode_fps)

        if self.step_count % 100 == 0:
            self._update_global_momentum()

        observations = list(self.world.get_observations(self.agent, self._obs_local_pts, self._agent_filter))
        self.last_observation =  np.expand_dims(np.array(observations, dtype=np.uint8), axis=0)

        reward = self.step_reward
        self.step_reward = 0
        self.step_count += 1

        if self.step_count % 100 == 0:
            self._regrow_food()

        return self.last_observation, reward, self.renderer.exited or False, None

    def _on_agent_rock_collision(self, arbiter, space, data)->bool:
        if arbiter.shapes:
            for shape in arbiter.shapes:
                if shape.collision_type == BoxWorldEnv.OBJ_TYPE_ROCK:
                    self._add_step_reward(-shape.body.mass * shape.body.velocity.get_length() / 100.0)

        return True
    def _on_agent_food_collision(self, arbiter, space, data)->bool:
        if arbiter.shapes:
            for shape in arbiter.shapes:
                if shape.collision_type == BoxWorldEnv.OBJ_TYPE_FOOD and shape not in self.eaten_foods:
                    self._add_step_reward(shape.body.mass)
                    self._set_eaten_status(shape, True)              

        return False        

    def _set_eaten_status(self, shape, is_eaten)->None:
        if is_eaten:
            shape.color = BoxWorldEnv.EATEN_FOOD_COLOR
            self.eaten_foods[shape] = self.step_count # record time step when food was eater
        else:
            shape.color = BoxWorldEnv.AVAIL_FOOD_COLOR
            self.eaten_foods.pop(shape, None)

    def _regrow_food(self):
        regrown = []
        for shape, eaten_step in self.eaten_foods.items():
            probability = (self.step_count-eaten_step)/shape.body.mass
            if random.random()*probability >= 1:
                regrown.append(shape)
        for shape in regrown:
            self._set_eaten_status(shape, False)

    def _add_step_reward(self, reward:float)->None:
        self.episod_reward += reward
        self.step_reward += reward

    def render(self, mode='human'):
        self.renderer.render(self.world, self.last_observation, self.episod_reward, self.last_total_momentum, mode=mode)

    def close(self):
        self.obss, self.foods = [], []
        if self.world:
            self.world.end()
        self.world = None

    def _get_random_circle_radius(self)->float:
        return max(5.0, abs(random.normalvariate(self.circle_radius, self.circle_radius/2)))

    def _get_random_box_size(self)->float:
        return (max(5.0, abs(random.normalvariate(self.box_size[0], self.box_size[0]/2))),
            max(5.0, abs(random.normalvariate(self.box_size[1], self.box_size[1]/2))))

    def _get_random_bar_size(self)->float:
        return (max(10.0, abs(random.normalvariate(self.bar_size[0], self.bar_size[0]/4))),
            max(10.0, abs(random.normalvariate(self.bar_size[1], self.bar_size[1]/8))))

    def _get_pts_on_circle(self, count:int, length:float)->Iterator[Tuple[float, float]]:
        start = 0
        step = 2 * math.pi / count
        for i in range(count):
            angle = start + step * i
            yield math.cos(angle) * length, math.sin(angle) * length

    def _get_random_position(self, clearance)->tuple:
        return random.randint(clearance, self.xmax-clearance), random.randint(clearance, self.ymax-clearance)

    def _update_global_momentum(self):
        self.last_total_momentum = self.world.get_total_momentum()
        if self.initial_total_momentum is None:
            self.initial_total_momentum = self.last_total_momentum
        else:
            decrease = self.initial_total_momentum - self.last_total_momentum
            if decrease > 0:
                print('decreased momentum', decrease)
                obj = random.choice(self.foods)
                obj._apply_gaussian_impulse(obj.size, obj.mass, obj.mass*100)
                obj = random.choice(self.obss)
                obj._apply_gaussian_impulse((obj.radius, obj.radius), obj.mass, obj.mass*100)

