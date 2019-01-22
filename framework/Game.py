from abc import ABC, abstractmethod
import  pygame
import numpy as np


class BaseGame(ABC):
    '''
    Pygame class to illustrate the system
    '''
    def __init__(self, world,  size = 1000,
                 background_color =(0,0,0), target_color = (255,0,0), target_diameter = 21,  randomise_dynamics = True):
        pygame.init()

        #System attributes
        self.world  = world
        self.randomise_dynamics = randomise_dynamics

        #Target attributes
        self.target_color= target_color
        self.target_diameter = target_diameter


        #Display attributes
        self.screen_dims = (size, size)
        self.screen = None
        self.background_color = background_color

        # Pygame attributes
        self.background = None
        self.clock = pygame.time.Clock()


    def discretize_position(self,x,y):
        return (int(x*self.screen_dims[0]),int(y*self.screen_dims[1]))

    def undiscretize_position(self,x,y):
        return np.array([x/self.screen_dims[0],y/self.screen_dims[1]])


    @abstractmethod
    def snap(self):
        '''
        Draw a new frame of the screen and flip the display.
        '''
        pass


    def optimal_controller(self, state, dynamics):
        '''
        Get the optimal control action given the state.
        '''

        return self.world.optimal_policy(state)

    def play_game(self, controller = None, game_type = 'manual', speed=None):
        '''
        Launch the game.
        In manual player clicks on the screen to reset the target position, and keep dynamics constant.
        In automatic reset the world  according to the world parameters.
        :param controller: The controller that takes in the current state and outputs the action. Default controller
        consists of the optimal policy.
        :param game_type: manual or automoatic. Default manual.
        :param speed: How many pygame ticks before each iteration.
        :return: N/A
        '''
        is_running = True
        if(self.screen is None):
            self.screen = pygame.display.set_mode(self.screen_dims)
            self.background = pygame.Surface(self.screen.get_size())
            self.background = self.background.convert()
            self.background = self.background.fill(self.background_color)


        if(controller == None):
            controller = self.optimal_controller

        while is_running:

            if(game_type == 'manual'):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        is_running = False
                        break
                    elif event.type == pygame.MOUSEBUTTONUP:
                        # Set a new target
                        target_pos = pygame.mouse.get_pos()
                        target_pos = self.undiscretize_position(*target_pos)
                        self.world.reset(reset_dynamics = self.randomise_dynamics)
                        self.world.update_target_pos(target_pos)
                        self.world.reset( from_state = self.world.state)
            else:
                self.world.reset(reset_dynamics=self.randomise_dynamics)

            self.snap()
            if (speed is not None):
                self.clock.tick(speed)

            while (not self.world.terminated):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        is_running = False
                        break

                action = controller(state =self.world.state,dynamics = self.world.dynamics)
                self.world.action_response(action)

                if(speed is not None):
                    self.clock.tick(speed)

                self.snap()

        pygame.quit()
        self.screen = None