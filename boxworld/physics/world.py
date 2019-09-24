from typing import Iterator, Callable, Any
import pymunk
import time
from .body import Body

class World:
    def __init__(self, xmax:int, ymax:int, gravityx:float=0.0, gravityy:float=0.0)->None:
        # pymonk has origin on left bottom
        self.xmax, self.ymax = xmax, ymax
        self.space = pymunk.Space()
        self.space.gravity = gravityx, gravityy
        self.last_step_time = time.time()
        self.color = (253, 223, 211, 0)
        self.collision_handlers = []

    def step(self, dt:float=None)->None:
        dt = dt or (time.time() - self.last_step_time)
        self.last_step_time = time.time()
        self.space.step(dt)

    def add(self, body:Body)->None:
        self.space.add(body.shape, body.body)

    def create_boundry(self, width=10, friction:float=0.0, elasticity:float=1.0, collision_type:int=None)->None:
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

    def get_observations(self, body:Body, local_pts:Iterator[tuple], filter:pymunk.ShapeFilter)->Iterator[tuple]:
        start_pt = body.body.position
        for local_pt in local_pts:
            end_pt = body.body.local_to_world(local_pt)
            result = self.space.segment_query_first(start_pt, end_pt, 0, filter)
            yield result.shape.color if result and result.shape else (0,0,0,0)

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
        self.space = None