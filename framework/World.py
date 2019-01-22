from abc import ABC, abstractmethod


class BaseWorld(ABC):
    def __init__(self, success_radius, max_number_of_steps, change_target_pos, change_start_config):
        '''
        Abstract World class : Any enviroment within the framework should inherit form that class.
        :param success_radius:[float between 0 and 1] The radius around the target position where the reaching is considered a success.
        :param max_number_of_steps: [int] Maximum number of allowed steps per episode before terminating with failure.
        :param change_target_pos: [bool] Whether to change the target position upon each reset
        :param change_start_config: [bool] whether to change the start configuration upon each reset
        '''

        #Whether to reset the target and the start configurations.
        self.change_target_pos = change_target_pos
        self.change_start_config = change_start_config

        # Successful termination parameters
        self.terminated = False
        self.success = False
        self.success_radius = success_radius
        self.max_number_of_steps = max_number_of_steps

        # Keep track of the trajectory for each episode and the number of current steps within the episode
        self.trajectory = []
        self.steps = 0

    @property
    @abstractmethod
    def state(self):
        '''
        Get the state of the world, the implementation must return a deep copy of the state.
        '''
        pass

    @state.setter
    @abstractmethod
    def state(self,val):
        '''
        Set the value of the state
        :param val: The value of the state
        '''
        pass

    @property
    @abstractmethod
    def state_size(self):
        '''
        Get the dimentionality of the state.
        '''
        pass

    @property
    @abstractmethod
    def action_size(self):
        '''
        Get the dimentionality of the action.
        '''
        pass


    @property
    @abstractmethod
    def dynamics(self):
        '''
        Get the dynamics parameters governing the agent in the world.
        '''
        pass

    @dynamics.setter
    @abstractmethod
    def dynamics(self,val):
        '''
        Set the dynamics parameters governing the agent in the world.
        '''
        pass

    @abstractmethod
    def update_target_pos(self,val):
        '''
        Update the target position and the state, taking into account the new target position
        '''

    @abstractmethod
    def optimal_policy(self):
        '''
        Return the optimal policy for the agent at a given time.
        '''
        pass


    @abstractmethod
    def reward(self):
        '''
        Return the reward at a given time.
        '''
        pass

    @abstractmethod
    def action_response(self):
        '''
        Govern the response of an agent action.
        :return: The new state, the reward and whether the episode terminated.
        '''
        pass

    @abstractmethod
    def reset(self):
        '''
        Reset the state of the world after the end of an episode.
        '''
        pass


    @abstractmethod
    def standardise(self, object, type):
        '''
        Standardise data to a certain range for better learning.
        '''

    @abstractmethod
    def un_standardise(self, object,type):
        '''
        Take standardised data and return the un-standardised version.
        '''