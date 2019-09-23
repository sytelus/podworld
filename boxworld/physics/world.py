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

    def step(self, dt:float=None)->None:
        dt = dt or (time.time() - self.last_step_time)
        self.last_step_time = time.time()
        self.space.step(dt)

    def add(self, body:Body)->None:
        self.space.add(body.shape, body.body)

    def create_boundry(self, width=10, friction:float=0.0, elasticity:float=1.0)->None:
        xmax, ymax = self.xmax, self.ymax
        walls= [pymunk.Segment(self.space.static_body, (-width, -width), (-width, ymax+width), width)
                    ,pymunk.Segment(self.space.static_body, (-width, ymax+width), (xmax+width, ymax+width), width)
                    ,pymunk.Segment(self.space.static_body, (xmax+width, ymax+width), (xmax+width, -width), width)
                    ,pymunk.Segment(self.space.static_body, (-width, -width), (xmax+width, -width), width)
                    ] 
        for s in walls:
            s.friction = friction
            s.elasticity = elasticity
            s.group = 1

        self.space.add(walls)