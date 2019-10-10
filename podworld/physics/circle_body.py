from .body import Body
import pymunk
import numpy as np

class CircleBody(Body):
    def __init__(self, mass:float, position, angle:float, radius:float, rgba_color:tuple=None, 
        friction:float=0.0, elasticity:float=1.0, category_mask:int=None, 
        collision_type:int=None, inertia=None, inner_radius=0)->None:

        super(CircleBody, self).__init__()

        self.radius, self.mass, self.inner_radius = radius, mass, inner_radius
        self.inertia = inertia or pymunk.moment_for_circle(mass, inner_radius, radius)
        
        self.body = pymunk.Body(self.mass, self.inertia)
        self.body.position = position
        self.body.angle = angle
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.friction = friction
        self.shape.elasticity = elasticity
        if category_mask:
            self.shape.filter = pymunk.ShapeFilter(categories=category_mask)
        if collision_type is not None:
            self.shape.collision_type = collision_type
        self.shape.color = rgba_color or tuple(np.random.randint(256, size=3)) + (255,)