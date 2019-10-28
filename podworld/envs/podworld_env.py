import gym
from gym import spaces, utils

from ..render.renderer import Renderer, RenderInfo 
from ..physics.world import Body
from ..physics.world import World
from ..physics.box_body import BoxBody
from ..physics.circle_body import CircleBody

import random
import math
import sys
import numpy as np
from typing import Iterator, Tuple
from gym.envs.registration import register

class PodWorldEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 60} #TODO: support 'ansi'

    OBJ_TYPE_ROCK = 1
    OBJ_TYPE_FOOD = 2
    OBJ_TYPE_BAR = 3
    OBJ_TYPE_WALL = 4
    OBJ_TYPE_AGENT = 0

    EATEN_FOOD_COLOR = (194, 199, 190, 255)
    AVAIL_FOOD_COLOR = (74, 198, 73, 255)
    OBS_COLOR = (162, 110, 180,255)
    NOOBJ_COLOR = (0,0,0,0)

    def __init__(self, food_count=10, obs_count=30, xmax=2560, ymax=1440, seed=42,
        box_size=(40.0, 40.0), circle_radius=30.0, box_mass=1.0, circle_mass=1.0,
        bar_count=4, bar_size=(900.0,40.0), bar_mass=10.0, 
        agent_radius=40.0, agent_mass=100.0, agent_obs_length=200.0, 
        agent_ray_count=64, agent_actuator_count=16,
        obs_start_angle=0, obs_end_angle=2 * math.pi, obs_mode=World.OBS_MODE_RGB,
        act_start_angle=0, act_end_angle=2 * math.pi,
        action_strength=2000.0, friction=0.1, elasticity=0.7,
        food_impulse=10.0, obs_impulse=10.0, bar_impulse=2.0, init_impulse_factor=1.0,
        max_steps=2**31-1, reward_factor=1.0)->None:

        self.seed(seed)

        self.obss, self.foods, self.bars, self.agent, self.world = \
            None, None, None, None, None
        self.eaten_foods = None
        self.step_reward, self.episod_reward, self.step_count = None, None, None
        self.initial_total_momentum, self.last_total_momentum = None, None
        self.last_thrust, self.last_action, self.sensor_probs = None, None, None

        
        self.food_count, self.obs_count, self.xmax, self.ymax = food_count, obs_count, xmax, ymax
        self.box_size, self.circle_radius, self.box_mass, self.circle_mass = box_size, circle_radius, box_mass, circle_mass
        self.bar_count, self.bar_size, self.bar_mass = bar_count, bar_size, bar_mass
        self.agent_radius, self.agent_mass, self.agent_obs_length = agent_radius, agent_mass, agent_obs_length
        self.agent_ray_count, self.action_strength = agent_ray_count, action_strength
        self.agent_actuator_count = agent_actuator_count
        self.obs_start_angle, self.obs_end_angle = obs_start_angle, obs_end_angle
        self.obs_mode = obs_mode
        self.act_start_angle, self.act_end_angle = act_start_angle, act_end_angle
        self.friction, self.elasticity = friction, elasticity
        self.food_impulse, self.obs_impulse, self.bar_impulse, self.init_impulse_factor = \
            food_impulse, obs_impulse, bar_impulse, init_impulse_factor
        self.max_steps, self.reward_factor = max_steps, reward_factor

        self.action_space = spaces.Discrete(agent_actuator_count+1) # directions clockwise, action 0 is no op
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, agent_ray_count, 3), dtype=np.uint8)

        self._obs_local_pts = list(self._get_pts_on_circle(
            self.observation_space.shape[1], self.agent_obs_length, obs_start_angle, obs_end_angle))
        self._action_pts = list(self._get_pts_on_circle(
            self.action_space.n-1, self.action_strength, act_start_angle, act_end_angle))
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
        self.world = World(name='PodWorld', xmax=self.xmax, ymax=self.ymax)
        self.world.create_boundry(collision_type=PodWorldEnv.OBJ_TYPE_WALL)
        self.world.set_collision_callback(PodWorldEnv.OBJ_TYPE_AGENT, PodWorldEnv.OBJ_TYPE_FOOD, 
            self._on_agent_food_collision)
        self.world.set_collision_callback(PodWorldEnv.OBJ_TYPE_AGENT, PodWorldEnv.OBJ_TYPE_ROCK, 
            self._on_agent_rock_collision)
        self.initial_total_momentum, self.last_total_momentum = None, None

        for _ in range(self.obs_count):
            radius = self._get_random_circle_radius()
            obj = CircleBody(mass=self.circle_mass * radius * radius, 
                position=self._get_random_position(clearance=10),
                angle=random.random() * 2 * math.pi, collision_type=PodWorldEnv.OBJ_TYPE_ROCK,
                radius=radius, rgba_color=PodWorldEnv.OBS_COLOR, 
                friction=self.friction, elasticity=self.elasticity)
            self.world.add(obj)
            self.obss.append(obj)            
            obj._apply_gaussian_impulse((radius, radius), obj.mass, obj.mass*self.obs_impulse*self.init_impulse_factor)
        for _ in range(self.food_count):
            box_size = self._get_random_box_size()
            mass = self.box_mass * box_size[0] * box_size[1]
            obj = BoxBody(mass=mass, 
                position=self._get_random_position(clearance=10),
                angle=random.random() * 2 * math.pi, collision_type=PodWorldEnv.OBJ_TYPE_FOOD,
                size=box_size, corner_radius=1.0, rgba_color=PodWorldEnv.AVAIL_FOOD_COLOR, 
                friction=self.friction, elasticity=self.elasticity)
            self.world.add(obj)
            self.foods.append(obj)
            obj._apply_gaussian_impulse(box_size, obj.mass, obj.mass*self.food_impulse*self.init_impulse_factor)
        for i in range(self.bar_count):
            bar_size = self._get_random_bar_size()
            obj = BoxBody(mass=self.bar_mass * bar_size[0] * bar_size[1], 
                position=(self.xmax/(self.bar_count+1)*(i+1), random.randint(0,self.ymax)),
                angle=random.random() * 2 * math.pi, collision_type=PodWorldEnv.OBJ_TYPE_BAR,
                size=bar_size, corner_radius=1.0, rgba_color= (232, 184, 164, 255), 
                friction=self.friction, elasticity=self.elasticity)
            self.world.add(obj)
            self.bars.append(obj)
            obj._apply_gaussian_impulse(bar_size, obj.mass, obj.mass*self.bar_impulse*self.init_impulse_factor)

        # add agent
        self._agent_filter = World.get_filter(self._agent_mask)        
        self.agent = CircleBody(mass=self.agent_mass, 
            position=self._get_random_position(clearance=self.agent_radius),
            angle=random.random() * 2 * math.pi, collision_type=PodWorldEnv.OBJ_TYPE_AGENT,
            radius=self.agent_radius, rgba_color=(224, 23, 33,255), category_mask=self._agent_mask, 
            friction=self.friction, elasticity=self.elasticity)
        self.world.add(self.agent, name='agent')

        self._update_observation()
        return self.last_observation

    # overriden method
    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def step(self, action):
        # perform action
        self.last_thrust = None
        if action > 0:
            self.last_thrust = self._action_pts[action-1]
            self.agent.apply_impulse(self.last_thrust, (0,0))

        self.world.step(1.0/self.renderer.human_mode_fps)

        if self.step_count % 100 == 0:
            self._update_global_momentum()

        self._update_observation()

        reward = self.step_reward
        self.step_reward = 0.0
        self.step_count += 1
        self.last_action = action

        if self.step_count % 100 == 0:
            self._regrow_food()

        done = self.renderer.exited or self.step_count >= self.max_steps

        return self.last_observation, reward/self.reward_factor, done, {}

    def _update_observation(self)->None:
        pixels = list(self.world.get_observations(
            self.agent, self._obs_local_pts, self._agent_filter, self.obs_mode))
        self.last_observation = np.expand_dims(np.array(pixels, dtype=np.uint8), axis=0)

    def _on_agent_rock_collision(self, arbiter, space, data)->bool:
        if arbiter.shapes:
            for shape in arbiter.shapes:
                if shape.collision_type == PodWorldEnv.OBJ_TYPE_ROCK:
                    self._add_step_reward( \
                        -shape.body.mass * shape.body.velocity.get_length() / 100.0)

        return True
    def _on_agent_food_collision(self, arbiter, space, data)->bool:
        if arbiter.shapes:
            for shape in arbiter.shapes:
                if shape.collision_type == PodWorldEnv.OBJ_TYPE_FOOD and \
                    shape not in self.eaten_foods:

                    self._add_step_reward(shape.body.mass)
                    self._set_eaten_status(shape, True)              

        return False        

    def _set_eaten_status(self, shape, is_eaten)->None:
        if is_eaten:
            shape.color = PodWorldEnv.EATEN_FOOD_COLOR
            self.eaten_foods[shape] = self.step_count # record time step when food was eater
        else:
            shape.color = PodWorldEnv.AVAIL_FOOD_COLOR
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
        render_info = RenderInfo(self.world, self.last_observation, self.episod_reward, 
            self.last_total_momentum, self.last_action, self.step_reward, self.last_thrust,
            self._obs_local_pts, self.sensor_probs, self.agent_obs_length)
        return self.renderer.render(render_info, mode=mode)

    def close(self):
        self.obss, self.foods = [], []
        self.renderer.close()
        if self.world:
            self.world.end()
        self.world = None

    def _get_random_circle_radius(self)->float:
        return max(5.0, abs(random.normalvariate(self.circle_radius, self.circle_radius/2)))

    def _get_random_box_size(self)->Tuple[float,float]:
        return (max(5.0, abs(random.normalvariate(self.box_size[0], self.box_size[0]/2))),
            max(5.0, abs(random.normalvariate(self.box_size[1], self.box_size[1]/2))))

    def _get_random_bar_size(self)->Tuple[float,float]:
        return (max(10.0, abs(random.normalvariate(self.bar_size[0], self.bar_size[0]/4))),
            max(10.0, abs(random.normalvariate(self.bar_size[1], self.bar_size[1]/8))))

    def _get_pts_on_circle(self, count:int, radius:float, start_angle:float, end_angle:float)->Iterator[Tuple[float, float]]:
        step = (end_angle-start_angle) / count
        for i in range(count):
            angle = start_angle + step * i
            yield math.cos(angle) * radius, math.sin(angle) * radius

    def _get_random_position(self, clearance)->tuple:
        return random.randint(clearance, self.xmax-clearance), random.randint(clearance, self.ymax-clearance)

    def _update_global_momentum(self):
        self.last_total_momentum = self.world.get_total_momentum()
        if self.initial_total_momentum is None:
            self.initial_total_momentum = self.last_total_momentum
        else:
            decrease = self.initial_total_momentum - self.last_total_momentum
            if decrease > 0:
                #print('decreased momentum', decrease)
                obj = random.choice(self.foods)
                obj._apply_gaussian_impulse(obj.size, obj.mass, obj.mass*100)
                obj = random.choice(self.obss)
                obj._apply_gaussian_impulse((obj.radius, obj.radius), obj.mass, obj.mass*100)

    def get_action_meanings(self):
        return ['No Thrust' if i==0 else 'Activate thruster {}'.format(i) \
             for i in range(self.action_space.n)]                

