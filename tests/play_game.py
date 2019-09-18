from boxworld.env import Env
import numpy as np
from pygame.locals import *
import pygame
import time
import math
import sys

x = np.random.randint(600, 800)
y = np.random.randint(50, 250)

agent_parameters = {
    'radius': 15,
    'speed': 10,
    'rotation_speed' : math.pi/8,
    'living_penalty': 0,
    'position': (x,y),
    'angle': 'random',
    'sensors': [
        {
           'nameSensor' : 'proximity_test',
           'typeSensor': 'proximity',
           'fovResolution': 64,
           'fovRange': 300,
           'fovAngle': math.pi ,
           'bodyAnchor': 'body',
           'd_r': 0,
           'd_theta': 0,
           'd_relativeOrientation': 0,
           'display': False,
        }
    ],
    'actions': ['forward', 'turn_left', 'turn_right', 'left', 'right', 'backward'],
    'measurements': ['health', 'poisons', 'fruits'],
    'texture': {
        'type': 'color',
        'c': (255, 255, 255)
    },
    'normalize_measurements': False,
    'normalize_states': False,
    'normalize_rewards': False
}

env_parameters = {
    'map':False,
    'n_rooms': 2,
    'display': True,
    'horizon': 10001,
    'shape': (900, 600),
    'mode': 'time',
    'poisons': {
        'number': 0,
        'positions': 'random',
        'size': 10,
        'reward': -10,
        'respawn': True,
        'texture': {
            'type': 'color',
            'c': (255, 255, 255),
        }
    },
    'fruits': {
        'number': 0,
        'positions': 'random',
        'size': 10,
        'reward': 10,
        'respawn': True,
        'texture': {
            'type': 'color',
            'c': (255, 150, 0),
        }
    },
    'obstacles': [
        {
            'shape': 'rectangle',
            'width': 50,
            'length': 300,
            'angle': 0,
            'position': (150, 150),
            'texture': {
                'type': 'color',
                'c': (100, 0, 0),
            }
        },
        {
            'shape': 'rectangle',
            'width': 50,
            'length': 300,
            'angle': math.pi/2,
            'position': (500, 500),
            'texture': {
                'type': 'color',
                'c': (50, 50, 0),
            }
        },
        {
            'shape': 'rectangle',
            'width': 50,
            'length': 500,
            'angle': 3*math.pi/4,
            'position': (350, 300),
            'texture': {
                'type': 'color',
                'c': (0, 0, 100),
            }
        },
        {
            'shape': 'circle',
            'position': (700, 300),
            'radius': 100,
            'texture': {
                'type': 'color',
                'c': (0, 50, 50),
            }
        }
    ],
    'walls_texture': {
        'type': 'color',
        'c': (50, 50, 50)
    },
    'agent': agent_parameters
}


"""
To play:
z = go forward
left arrow = turn left
right arrow = turn right

For it to work, you still have to have the game display ON, and this should be the active window.
If you want to play in partially observable conditions, just don't look at this display window :)
"""

env = Env(**env_parameters)
n = len(agent_parameters['actions'])
meas, sens = None, None
action_forward = {'longitudinal_velocity':0, 'lateral_velocity':1, 'angular_velocity':0}
action_turn_left = {'longitudinal_velocity':0, 'lateral_velocity':0, 'angular_velocity':1}
action_turn_right = {'longitudinal_velocity':0, 'lateral_velocity':1, 'angular_velocity':-1}

pygame.init()

start = time.time()
done = False
for i in range(5):
    time.sleep(1)
    while not done:
        env.reload_screen()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_z:
                    sens, r, done, meas = env.step(action_forward)
                    print(meas)
                    print(r)
                if event.key == K_LEFT:
                    sens, r, done, meas = env.step(action_turn_left)
                    print(meas)
                    print(r)
                if event.key == K_RIGHT:
                    sens, r, done, meas = env.step(action_turn_right)
                    print(meas)
                    print(r)
            if done:
                break

    env.reset()
    done = False
end = time.time()

print(end - start)


