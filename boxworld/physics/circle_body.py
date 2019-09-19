from .body import Body
import pymunk

class CircleBody(Body):
    def __init__(self, mass:float, position, radius:float, friction:float=0.0, elasticity:float=1.0,
     inertia=None, inner_radius=0)->None:

        super(CircleBody, self).__init__()

        self.radius, self.mass, self.inner_radius = radius, mass, inner_radius
        self.inertia = inertia or pymunk.moment_for_circle(mass, inner_radius, radius)
        
        self.body = pymunk.Body(self.mass, self.inertia)
        self.body.position = position
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.friction = friction
        self.shape.elasticity = elasticity