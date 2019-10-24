from typing import Iterator, Callable, Any, Dict
import pymunk
import time
import numpy as np
from .body import Body

class World:
    OBS_MODE_RGB='RGB'
    OBS_MODE_RGBA='RGBA'
    OBS_MODE_RGBD='RGBD'
    OBS_MODE_RGBAD='RGBAD'

    def __init__(self, name:str, xmax:int, ymax:int, gravityx:float=0.0, gravityy:float=0.0)->None:
        # pymonk has origin on left bottom
        self.name = name
        self.xmax, self.ymax = xmax, ymax
        self.space = pymunk.Space()
        self.space.gravity = gravityx, gravityy
        self.last_step_time = time.time()
        self.color = (253, 223, 211, 0)
        self.collision_handlers = []
        self.named_bodies:Dict[str, Body] = {}

    def step(self, dt:float=None)->None:
        dt = dt or (time.time() - self.last_step_time)
        self.last_step_time = time.time()
        self.space.step(dt)

    def add(self, body:Body, name=None)->None:
        self.space.add(body.shape, body.body)
        if name is not None:
            self.named_bodies[name] = body

    def create_boundry(self, width=100, friction:float=0.0, elasticity:float=1.0, collision_type:int=None)->None:
        xmax, ymax = self.xmax, self.ymax
        walls= [pymunk.Segment(self.space.static_body, (-width, -width), (-width, ymax+width), width)
                    ,pymunk.Segment(self.space.static_body, (-width, ymax+width), (xmax+width, ymax+width), width)
                    ,pymunk.Segment(self.space.static_body, (xmax+width, ymax+width), (xmax+width, -width), width)
                    ,pymunk.Segment(self.space.static_body, (-width, -width), (xmax+width, -width), width)
                    ] 
        for s in walls:
            s.friction = friction
            s.elasticity = elasticity
            s.color = (247, 207, 49, 255)
            if collision_type is not None:
                s.collision_type = collision_type

        self.space.add(walls)

    @staticmethod
    def get_filter(value:int)->pymunk.ShapeFilter:
        return pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS ^ value)

    # for each pixel, return RGBA tuple
    def get_observations(self, body:Body, local_pts:Iterator[tuple], 
        filter:pymunk.ShapeFilter, obs_mode=OBS_MODE_RGB)->Iterator[tuple]:

        start_pt = body.body.position
        for local_pt in local_pts:
            end_pt = body.body.local_to_world(local_pt)
            result = self.space.segment_query_first(start_pt, end_pt, 0, filter)
            color, pos = (0,0,0,0), None
            if result and result.shape:
                color = result.shape.color
                pos = result.shape.body.position

            if obs_mode == World.OBS_MODE_RGB or obs_mode == World.OBS_MODE_RGBD:
                color = color[:3]
            if obs_mode == World.OBS_MODE_RGBAD or obs_mode == World.OBS_MODE_RGBD:
                dist = 1.0
                if pos is not None:
                    dist = np.linalg.norm([body.body.position[0]-pos[0], 
                        body.body.position[1]-pos[1]])
                color += (dist,)
            #TODO: check invalid modes
            yield color

    def set_collision_callback(self, collision_type_a:int, collision_type_b:int, 
        callback:Callable[[pymunk.Arbiter, pymunk.Space, Any],bool], at_begin=True)->None:

        ch = self.space.add_collision_handler(collision_type_a, collision_type_b)
        if at_begin:
            ch.begin = callback
        else:
            ch.end = callback
        self.collision_handlers.append(ch)

    def end(self):
        self.collision_handlers.clear()
        self.named_bodies.clear()
        self.space = None

    def get_total_momentum(self):
        total_momentum = 0.0
        for body in self.space.bodies:
            if body.body_type == pymunk.Body.DYNAMIC:
                total_momentum += body.mass * body.velocity.get_length()
        return total_momentum