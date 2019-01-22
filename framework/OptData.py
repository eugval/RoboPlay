from torch.utils.data import Dataset
import torch
import numpy as np

class OptData(Dataset):
    def __init__(self,data_buffer, utils, state_sequence_length, test= False):
        self.data_buffer = data_buffer
        self.utils = utils
        self.standardise_data = utils.standardise_data
        self.test = test
        self.state_sequence_length = state_sequence_length
        self.input_size = self.utils.world.state_size + \
        (state_sequence_length-1)*( self.utils.world.state_size + self.utils.world.action_size)

    def __len__(self):
        return self.data_buffer.states_len(self.test)

    def __getitem__(self,idx):
        state_t = self.data_buffer.get_state(idx,self.test)
        state_sequence = np.copy(state_t)

        state_sequence = self.data_buffer.expand_inputs_for_training(state_sequence,idx,self.utils.world,
                                                                     self.utils.standardise_data,self.state_sequence_length,
                                                                     self.input_size,self.test)

        if(self.standardise_data):
            state_t = self.utils.world.standardise(state_t,'state')


        dynamics_parameters = self.data_buffer.get_dynamics_params(idx,self.test)
        optimal_action_t = self.data_buffer.get_optimal_action(idx,self.test)

        if(self.standardise_data):
            dynamics_parameters = self.utils.world.standardise(dynamics_parameters,'dynamics')
            optimal_action_t = self.utils.world.standardise(optimal_action_t,'action')

        sample = {'state_t':torch.from_numpy(state_t),
                  'state_sequence': torch.from_numpy(state_sequence),
                  'dynamics_parameters':torch.from_numpy(dynamics_parameters),
                  'optimal_action_t':torch.from_numpy(optimal_action_t)
                  }

        return sample


    def verify_dimentionality(self, model):
        if(self.input_size != model.input_size):
            raise ValueError('Training and model input sizes differ.')

