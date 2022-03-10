import numpy as np
import pygame

class Pygame_2d:
    def __init__(self, im_width, im_height):
        # Initialize pygame environment
        pygame.init()
        self.display = pygame.display.set_mode((im_width, im_height))
        pygame.display.set_caption("gym_pygame")
        self.clock = pygame.time.Clock()

        # Fill pygame window with black background
        self.display.fill((0,0,0))
        pygame.display.flip()

        # TOGGLES
        self.quit            = False
        self.manual          = False       # To enable manual control
        self.autopilot       = False       # To set autopilot for immitation

    def to_quit(self):
        """Check if pygame window terminated"""
        return self.quit
    
    def set_autopilot(self):
        """Returns true if autopilot toggle enabled"""
        return self.autopilot
    
    def get_actions(self, action_space):
        """Returns manual actions to execute"""
        # Zero placeholder value
        action = np.zeros(action_space.shape)      # not sure if .shape works with gym.spaces.Discrete
        return action
    
    def render(self,image):
        self.clock.tick()
        self.display.fill((0,0,0))
        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        self.display.blit(surface, (0,0))
        pygame.display.flip()          # Update display/game window
    
    def event_parser(self):
        # This is event parser module

        # TODO 
        # Add manual control toggle
        #            - seperate keys to enable or disable
        #            - disable autopilot toggle if this is enabled
        # Add manual control events
        # Add events to change sensor value being displayed

        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                self.quit = True
                pygame.quit()
            elif event.type==pygame.KEYUP:

                # Enable or disable autopilot
                if event.key==pygame.K_COMMA:
                    if not self.autopilot:
                        self.autopilot = True
                        self.manual    = False
                        print('Autopilot Enabled')
                    else:
                        self.autopilot = False
                        print('Autopilot Disabled')
