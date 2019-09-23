import sys
import numpy as np
import pygame
import pymunk.pygame_util

from ..physics.world import World

class Renderer:
    def __init__(self, viewport_size:tuple=None, human_mode_fps=60)->None:
        self.screen = None
        self.human_mode_fps = human_mode_fps
        self.viewport_size = viewport_size
        self.exited = False

    def _init_lazy_render(self, world:World):
        if not self.screen:
            pygame.init()
            self.viewport_size = self.viewport_size or (world.xmax, world.ymax)
            self.screen = pygame.display.set_mode(self.viewport_size)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 16)

    def render(self, world:World, mode='human')->np.ndarray:
        self._init_lazy_render(world)
        self._handle_player_events()
        if self.exited:
            return None        
        self._draw_world(world)

        ret:np.ndarray = None
        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.human_mode_fps)
        else:
            self.clock.tick() # don't introduce additional delay for human vision
            ret = np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

        return ret

    def _handle_player_events(self)->None:
        if self.exited:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.exited = True # sys.exit()   
            elif event.type == pygame.VIDEORESIZE:
                self.viewport_size = event.size
                screen = pygame.display.set_mode(self.viewport_size)
                print('Viewport size changed to', video_size)                

    def _draw_world(self, world:World)->None:
        self.screen.fill(pygame.color.THECOLORS["white"])
        options = pymunk.pygame_util.DrawOptions(self.screen)
        world.space.debug_draw(options)