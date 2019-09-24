import sys
import platform
import numpy as np
import pygame
import pymunk.pygame_util

from ..physics.world import World

class Renderer:
    def __init__(self, viewport_size:tuple=None, human_mode_fps=60, info_height=None)->None:
        self.screen:pygame.Surface = None
        self.human_mode_fps = human_mode_fps
        self.viewport_size = viewport_size
        self.exited = False
        self.info_height = info_height or 100
        self.key_map = {pygame.K_RIGHT:1, pygame.K_UP:0, pygame.K_LEFT:3, pygame.K_DOWN:2}
        self.last_mapped_key:int = None

    def _init_lazy_render(self, world:World):
        if not self.screen:
            if platform.system() == 'Windows':
                # don't scale window size for DPI scaling
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()            
            pygame.init()
            self.viewport_size = self.viewport_size or (world.xmax, world.ymax)
            self.viewport_size = self.viewport_size[0], self.viewport_size[1] + self.info_height
            self.screen = pygame.display.set_mode(self.viewport_size)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(pygame.font.get_default_font(), 36)

    def render(self, world:World, last_observation:np.ndarray, episod_reward:float, 
        total_momentum:float, mode='human')->np.ndarray:

        self._init_lazy_render(world)
        self._handle_player_events()
        if self.exited:
            return None        
        self._draw_world(world)
        self._draw_info(last_observation, episod_reward, total_momentum)

        ret:np.ndarray = None
        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.human_mode_fps)
        else:
            self.clock.tick() # don't introduce additional delay for human vision
            ret = np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

        return ret

    def _handle_player_events(self)->None:
        self.last_mapped_key = None

        if self.exited:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
                (event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE):

                pygame.quit()
                self.exited = True # sys.exit()   
            elif event.type == pygame.VIDEORESIZE:
                self.viewport_size = event.size
                self.screen = pygame.display.set_mode(self.viewport_size)
                print('Viewport size changed to', self.viewport_size) 
            elif event.type == pygame.KEYDOWN:
                self.last_mapped_key = self.key_map.get(event.key, None)               

    def _draw_info(self, last_observation:np.ndarray, episod_reward:float, total_momentum:float):
        width = self.screen.get_width()
        obs_height = self.info_height / 2.0       
        self.screen.fill(pygame.color.THECOLORS['white'], 
            pygame.rect.Rect(0, 0, width, self.info_height))

        info_text = 'Reward: {:.2f}                                    Global Energy: {:.2f}'.format(episod_reward, total_momentum)
        text_surface = self.font.render(info_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (0,0)) 

        if last_observation is not None:
            obs_width = width / last_observation.shape[1]
            for i in range(last_observation.shape[1]):
                color = tuple(last_observation[0, i])
                self.screen.fill(color, 
                            pygame.rect.Rect(i*obs_width, self.info_height-obs_height, obs_width, obs_height))

    def _draw_world(self, world:World)->None:
        self.screen.fill(world.color)
        options = pymunk.pygame_util.DrawOptions(self.screen)
        options.positive_y_is_up = False
        world.space.debug_draw(options)
