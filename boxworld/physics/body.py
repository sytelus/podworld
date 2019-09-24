import random
from abc import ABCMeta, abstractmethod
import pymunk
from pymunk import Vec2d

class Body(metaclass=ABCMeta):
    def __init__(self)->None:
        self.body:pymunk.Body = None
        self.shape:pymunk.Shape = None

    def apply_impulse(self, impulse:tuple, local_pos:tuple)->None:
        self.body.apply_impulse_at_local_point(impulse, local_pos)
    def apply_force(self, impulse:tuple, local_pos:tuple)->None:
        self.body.apply_force_at_local_point(impulse, local_pos)

    def _apply_gaussian_impulse(self, extent:tuple, magnitude_mean:float, magnitude_stddev:float)->None:
        x, y = extent
        impulse_pos = random.uniform(-x/2, x/2),random.uniform(-y/2, y/2)
        self.apply_impulse(impulse=Body._get_gaussian_impulse(magnitude_mean, magnitude_stddev), 
            local_pos=impulse_pos)

    @staticmethod
    def _get_gaussian_impulse(magnitude_mean:float, magnitude_stddev:float)->tuple:
        direction = Vec2d(random.uniform(-1, 1),random.uniform(-1, 1)).normalized()
        magnitude = random.normalvariate(magnitude_mean, magnitude_stddev)
        return tuple(magnitude * direction)