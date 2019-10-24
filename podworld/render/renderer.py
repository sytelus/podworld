import sys
import platform
import numpy as np
import pygame
import os
import pymunk.pygame_util
from typing import Tuple
from ..physics.world import World

class RenderInfo:
    def __init__(self, world:World, last_observation:np.ndarray, 
        episod_reward:float, total_momentum:float, action:int, reward:float,
        thrust_pt:Tuple[float, float], sensor_pts, sensor_probs, agent_obs_length)->None:

        self.world = world
        self.last_observation = last_observation
        self.episod_reward = episod_reward
        self.total_momentum = total_momentum
        self.action = action
        self.reward = reward   
        self.thrust_pt = thrust_pt
        self.sensor_pts = sensor_pts
        self.sensor_probs = sensor_probs
        self.agent_obs_length = agent_obs_length

class Renderer:
    def __init__(self, viewport_size:tuple=None, human_mode_fps=60, info_height=None)->None:
        self.screen:pygame.Surface = None
        self.human_mode_fps = human_mode_fps
        self.viewport_size = viewport_size
        self.exited = False
        self.info_height = info_height or 100
        self.key_map = {pygame.K_RIGHT:1, pygame.K_UP:0, pygame.K_LEFT:3, pygame.K_DOWN:2}
        self.last_mapped_key:int = None

    def _init_lazy_render(self, world:World, mode):
        if not self.screen:
            if mode != 'human':
                # set SDL to use the dummy NULL video driver, 
                # so it doesn't need a windowing system.
                os.environ["SDL_VIDEODRIVER"] = "dummy"
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
            pygame.display.set_caption('{}'.format(world.name or 'PyGame World'))

    def render(self, render_info:RenderInfo, mode='human')->np.ndarray:
        self._init_lazy_render(render_info.world, mode)
        self._handle_player_events()
        if self.exited:
            return None        
        self._draw_world(render_info.world)
        self._draw_info(render_info.last_observation, 
            render_info.episod_reward, render_info.total_momentum)
        self._draw_debug(render_info)

        ret:np.ndarray = None
        pygame.display.flip()
        ret = np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

        if mode == 'human':
            self.clock.tick(self.human_mode_fps)
        else:
            self.clock.tick() # don't introduce additional delay for human vision

        return ret

    def _draw_debug(self, render_info:RenderInfo)->None:
        agent = render_info.world.named_bodies.get('agent', None)
        if agent is not None:
            reward_color = None
            if render_info.reward < 0:
                reward_color = (100, 0, 0)
            elif render_info.reward > 0:
                reward_color = (50, 255, 50)

            pos_o = agent.body.position
            pos = (int(pos_o[0]), 
                    int(render_info.world.ymax-pos_o[1]+self.info_height))

            pygame.draw.circle(self.screen, (100,100,100), pos, 
                int(render_info.agent_obs_length), 1)

            if reward_color is not None:
                pygame.draw.circle(self.screen, reward_color, pos, 10, 0)
            if render_info.thrust_pt is not None:
                t = self.local2world_pt(pos_o, render_info.thrust_pt, 
                    agent.shape.radius, render_info.world)
                pygame.draw.line(self.screen, (100,100,255), 
                    pos, t, 10)
            # if render_info.sensor_probs is not None and render_info.sensor_pts is not None:
            #     for i,sensor_pt in enumerate(render_info.sensor_pts):
            #         p = self.local2world_pt(pos_o, sensor_pt, agent.shape.radius, render_info.world)
            #         c = int(render_info.sensor_probs[i] * 255)
            #         c = (255-c,255-c,0)
            #         pygame.draw.circle(self.screen, c, p, 10, 0)

    def local2world_pt(self, pos_o:Tuple[float, float], l_pt:Tuple[float, float], 
        clip:float, world)->Tuple[float, float]:

        t_o = list(l_pt)
        t_o = np.clip(t_o, -clip, clip)
        t_w = pos_o[0]+t_o[0], pos_o[1]+t_o[1]
        return int(t_w[0]), int(world.ymax-t_w[1]+self.info_height)

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

        info_text = 'Reward: {:.2f}                                    Global Energy: {:.2f}' \
            .format(episod_reward or 0.0, total_momentum or 0.0)
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

    def close(self):
        if self.screen:
            pygame.display.quit()
            self.screen = None
