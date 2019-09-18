import pygame
from pygame.locals import *
from pygame.color import *

import pymunk
from pymunk import Vec2d
import random

X,Y = 0,1
xmax,ymax = 690,600
ballr=5
ballm, linem=10, 10
wallw, linew=10, 10

def flipy(y):
    """Small hack to convert chipmunk physics to pygame coordinates"""
    return ymax-y

def main():
            
    pygame.init()
    screen = pygame.display.set_mode((xmax,ymax))
    clock = pygame.time.Clock()
    running = True
    
    ### Physics stuff
    space = pymunk.Space()
    space.gravity = 0.0, 0.0
    
    # walls - the left-top-right walls
    walls= [pymunk.Segment(space.static_body, (-wallw, -wallw), (-wallw, ymax+wallw), wallw)
                ,pymunk.Segment(space.static_body, (-wallw, ymax+wallw), (xmax+wallw, ymax+wallw), wallw)
                ,pymunk.Segment(space.static_body, (xmax+wallw, ymax+wallw), (xmax+wallw, -wallw), wallw)
                ,pymunk.Segment(space.static_body, (-wallw, -wallw), (xmax, -wallw), wallw)
                ] 
    for s in walls:
        s.friction = 0.0
        s.elasticity = 1.0
        s.group = 1

    space.add(walls)

    ## Balls
    balls = []
    lines = []
    run_physics = True

    for i in range(10):
        p = random.randint(ballr,xmax-ballr), random.randint(ballr,ymax-ballr)
        inertia = pymunk.moment_for_circle(ballm, 0, ballr, (0, 0))
        body = pymunk.Body(ballm, inertia)
        body.position = p
        shape = pymunk.Circle(body, ballr, (0,0))
        shape.friction = 0.0
        shape.elasticity = 1.0
        space.add(body, shape)
        power = random.randint(10,1000) * 3
        impulse = power * Vec2d(random.uniform(-1, 1),random.uniform(-1, 1))        
        body.apply_impulse_at_world_point(impulse, body.position)
        balls.append(shape)

    for i in range(3):
        length = int(random.uniform(0,0.5) * min(xmax, ymax))
        p1 = Vec2d(random.randint(length,xmax-length), random.randint(length,ymax-length))
        p2 = p1 + Vec2d(random.uniform(-1, 1),random.uniform(-1, 1)).normalized() * length
        inertia = pymunk.moment_for_segment(linem, p1, p2, linew)
        body = pymunk.Body(linem, inertia)
        shape= pymunk.Segment(body, p1, p2, linew)    
        shape.friction = 0.0
        shape.elasticity = 1.0
        space.add(body, shape)
        power = random.randint(10,1000) * 3
        impulse = power * Vec2d(random.uniform(-1, 1),random.uniform(-1, 1))       
        body.apply_impulse_at_world_point(impulse, shape.a + (shape.b-shape.a)*random.uniform(0, 1))
        lines.append(shape)

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False
            elif event.type == KEYDOWN and event.key == K_p:
                pass
                #pygame.image.save(screen, "balls_and_lines.png")        
            elif event.type == KEYDOWN and event.key == K_SPACE:    
                run_physics = not run_physics
       
        ### Update physics
        if run_physics:
            dt = 1.0/60.0
            for x in range(1):
                space.step(dt)
            
        ### Draw stuff
        screen.fill(THECOLORS["white"])

        for ball in balls:           
            r = ball.radius
            v = ball.body.position
            rot = ball.body.rotation_vector
            p = int(v.x), int(flipy(v.y))
            p2 = Vec2d(rot.x, -rot.y) * r * 0.9
            pygame.draw.circle(screen, THECOLORS["blue"], p, int(r), 0)
            pygame.draw.line(screen, THECOLORS["red"], p, p+p2)

        for line in lines:
            body = line.body   
            pv1 = body.position + line.a.rotated(body.angle)
            pv2 = body.position + line.b.rotated(body.angle)
            p1 = pv1.x, flipy(pv1.y)
            p2 = pv2.x, flipy(pv2.y)
            pygame.draw.line(screen, THECOLORS["orange"], p1, p2, linew)            

        ### Flip screen
        pygame.display.flip()
        clock.tick(50)
        pygame.display.set_caption("fps: " + str(clock.get_fps()))
        
if __name__ == '__main__':
    doprof = 0
    if not doprof: 
        main()
    else:
        import cProfile, pstats
        
        prof = cProfile.run("main()", "profile.prof")
        stats = pstats.Stats("profile.prof")
        stats.strip_dirs()
        stats.sort_stats('cumulative', 'time', 'calls')
        stats.print_stats(30)