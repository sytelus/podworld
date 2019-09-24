from .body import Body
import pymunk
import numpy as np

class BoxBody(Body):
    def __init__(self, mass:float, position, angle:float, size:tuple, rgba_color:tuple=None, 
        corner_radius:float=0.0, friction:float=0.0, elasticity:float=1.0, category_mask=None,
        collision_type:int=None, inertia=None, inner_radius=0)->None:

        super(BoxBody, self).__init__()
        
        self.size, self.corner_radius, self.mass = size, corner_radius, mass
        self.inertia = inertia or pymunk.moment_for_box(mass, size)
        
        self.body = pymunk.Body(self.mass, self.inertia)
        self.body.position = position
        self.body.angle = angle
        self.shape = pymunk.Poly.create_box(self.body, size, corner_radius)
        self.shape.friction = friction
        self.shape.elasticity = elasticity
        if category_mask:
            self.shape.filter = pymunk.ShapeFilter(categories=category_mask)
        if collision_type is not None:
            self.shape.collision_type = collision_type            
        self.shape.color = rgba_color or tuple(np.random.randint(256, size=3)) + (255,)