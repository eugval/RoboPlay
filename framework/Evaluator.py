from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    '''
    Class to create plots for evaluating the models.
    '''
    def __init__(self, data_buffer, model_folder):
        self.model_folder = model_folder
        self.data_buffer = data_buffer

    @abstractmethod
    def plot_training_history(self):
        '''
        Method to plot the training history as recorded by the training class.
        '''
        pass

    @abstractmethod
    def plot_states(self):
        '''
        Plot the states present in the buffer
        '''

    def change_model_folder(self, new_path):
        self.model_folder = new_path