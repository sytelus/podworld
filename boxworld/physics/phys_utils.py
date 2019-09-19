import pymunk


    def _get_random_impulse(self, magnitude)->tuple:
        direction = Vec2d(random.uniform(-1, 1),random.uniform(-1, 1)) 