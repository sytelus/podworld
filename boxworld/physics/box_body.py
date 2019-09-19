from .body import Body
import pymunk

class BoxBody(Body):
    def __init__(self, mass:float, position, size:tuple, corner_radius:float=0.0, friction:float=0.0, elasticity:float=1.0,
     inertia=None, inner_radius=0)->None:

        super(BoxBody, self).__init__()
        
        self.size, self.corner_radius, self.mass = size, corner_radius, mass
        self.inertia = inertia or pymunk.moment_for_box(mass, size)
        
        self.body = pymunk.Body(self.mass, self.inertia)
        self.body.position = position
        self.shape = pymunk.Poly.create_box(self.body, size, corner_radius)
        self.shape.friction = friction
        self.shape.elasticity = elasticity