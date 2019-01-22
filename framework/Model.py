from abc import ABC
import torch
import numpy as np

class Model(ABC):
    def __init__(self, network, device):
        self.network = network.to(device).double()
        self.device = device
        self.input_size = self.network.input_size
        self.output_size = self.network.output_size

    def infer(self,state):
        if (isinstance(state, torch.Tensor)):
            state = state.to(self.device)
        elif (isinstance(state, np.ndarray)):
            state = torch.from_numpy(state).to(self.device)

        if (len(state.size()) < 2):
            state = state.unsqueeze(0)

        with torch.no_grad():
            self.network.eval()
            out = self.network(state)
            self.network.train()

        out = torch.squeeze(out).cpu().numpy()
        return out


    def expand_input_for_inference(self, state, world, standardise, total_states = 1, append_dynamics = None,
                                   standardise_dynamics = False):
        count = total_states-1
        trajectory_index = 1

        if (standardise):
            state = world.standardise(state, 'state')

        while (count > 0 and trajectory_index <= len(world.trajectory)):
            state_action_pair = world.trajectory[-trajectory_index]
            state_only = state_action_pair[:world.state_size]
            action_only = state_action_pair[-world.action_size:]
            if (standardise):
                state_only = world.standardise(state_only, 'state')
                action_only = world.standardise(action_only, 'action')
            state = np.concatenate([state_only, action_only, state], axis=0)
            count -= 1
            trajectory_index += 1

        if (append_dynamics is not None):
            dyns = append_dynamics

            if (standardise_dynamics):
                dyns = world.standardise(dyns, 'dynamics')

            state = np.concatenate([dyns, state], axis=0)


        if (state.size < self.input_size):
            zeros_array = np.zeros(self.input_size)
            zeros_array[-state.size:] = state
            state = zeros_array

        if(state.size >  self.input_size):
            raise ValueError('Inputs have the wrong dimensions.')

        return state