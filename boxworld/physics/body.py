from abc import ABCMeta, abstractmethod
import pymunk

class Body(metaclass=ABCMeta):
    def __init__(self)->None:
        self.body:pymunk.Body = None
        self.shape:pymunk.Shape = None

    def apply_impulse(self, impulse:tuple, local_pos:tuple)->None:
        self.body.apply_impulse_at_local_point(impulse, local_pos)
