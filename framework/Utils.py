import os
from abc import ABC
import math
import numpy as np
import torch
import pickle


class Utils(ABC):
    def __init__(self, world, standardise_data):
        '''
        The Utils class give access to the world, the agent, and includes methods helpful at multiple
        levels of the system pipeline (training, testing, playing ect)
        '''
        self.world = world
        #self.agent = agent
        self.standardise_data = standardise_data

    def do_rollouts(self, inference_function, number_of_trials = 200, report_trajectories= False, initial_configs = None,
                      subset_proportion = 0.1, randomise_dynamics = False):
        '''
        Rollout a policy and record the accuracy (number of successful rollouts), and the average minimum cost reached
        with respect to the goal.
        Optionally also report the whole trajectory data for DAGGER rollouts.
        Optionally accept a set of initial conditions as the rollout initialisations.
        :param inference_function: Function that accepts the current state and dynamics and outputs and action.
        :param number_of_trials: [int] How many rollouts to do
        :param report_trajectories: [bool] report the full trajectories from the rollouts.
        :param initial_configs: [list of (state,dynamics) pairs] If present, sample from this list to set the initial
        conditions of each rollout.
        :param subset_proportion: [float between 0 and 1] If an initial configs list is provided, only select this
        subproportion of the list to make the rollouts.
        :param randomise_dynamics: [bool] Whether to randomise dynamics at each rollout.
        :return:
        Accuracy of the rollouts and the average minimum average cost achieved.
        Or
        Accuracy, ave min cost, a list of states visited, a list of actions taken, a list of dynamics at play and a list
        of trajectory indices (information necessary to add new data to the data buffer).
        '''
        print('Doing  rollouts..')
        success = 0
        cum_min_distance = 0
        states = []
        actions = []
        dynamics = []
        trajectory_indices = []


        # If initial configs is provided, the number of trials is equal to the size of this list times the proportion
        # of it to consider.
        if(initial_configs is None):
            initial_configs = []

        given_states_len = len(initial_configs)
        if(given_states_len>0):
            # Considered initial states are picked at random from the initial configs list.
            number_of_trials = math.ceil(subset_proportion*given_states_len)
            indx_array = np.random.choice(given_states_len,number_of_trials)


        for trial in range(number_of_trials):

            # Reset the world
            if(given_states_len>0):
                idx = indx_array[trial]
                self.world.reset(from_state = initial_configs[idx][0], specific_dynamics = initial_configs[idx][1])
            else:
                self.world.reset(reset_dynamics= randomise_dynamics)


            traj_idx = 0
            min_dist = float('inf')

            # Do rollouts
            while(not self.world.terminated):

                # Record trajectory snap
                state = self.world.state
                dyn = self.world.dynamics
                states.append(state)
                dynamics.append(dyn)
                trajectory_indices.append(traj_idx)

                # Do the inference
                action = inference_function(state = state, dynamics =dyn)

                if (self.standardise_data):
                    action = self.world.un_standardise(action, 'action')

                actions.append(action)

                # Update the world
                new_state, reward , terminated = self.world.action_response(action)

                # Record the minimum cost
                dist = -reward
                if(dist<min_dist):
                    min_dist=dist

                # Record the accuracy
                if(terminated and self.world.success):
                    success +=1

                traj_idx+=1

            cum_min_distance += min_dist

        # Calculate averages
        accuracy = success/number_of_trials
        average_min_dist = cum_min_distance/number_of_trials

        if(report_trajectories):
            return accuracy, average_min_dist, states, actions, dynamics, trajectory_indices

        return accuracy, average_min_dist



    def load_best_model(self, directory_path, prefix, criterion = 'test_soft_accuracy', find ='minimum'):
        '''
        Load the best model saved on some directory. Look at the training history with respect to some criterion to
        determine the best performing model.
        :param directory_path: [path] The directory the different models are saved.
        :param prefix: [string] Prefix for the model file name.
        :param criterion: [string] one of ['loss', 'test_accuracy', 'test_soft_accuracy', 'train_accuracy','train_soft_accuracy'].
        :param find: [string] One of ['minimum','maximum']. Whether to look for the minimum or maximum w.r.t the criterion.
        :return: The best model with respect to the criterion and the find value.
        '''
        his = pickle.load(open(os.path.join(directory_path, '{}_training_history.pckl'.format(prefix)), 'rb'))

        if (find == 'minimum'):
            best_val = float('inf')
            best_low = True
        elif(find == 'maximum'):
            best_val = -float('inf')
            best_low = False
        else:
            raise ValueError('Can only look for the minimum or maximum.')

        for e, val in enumerate(his[criterion]):
            if (best_low):
                if (val <= best_val):
                    best_val = val
                    best_checkpoint = e
            else:
                if (val >= best_val):
                    best_val = val
                    best_checkpoint = e

        best_model_path = os.path.join(directory_path, '{}_{}.pth'.format(directory_path, best_checkpoint))
        print('best model: {}, best_value: {}'.format(best_checkpoint, best_val))
        return self.load_model(best_model_path), best_model_path



    def load_model(self, path):
        '''
        Load a specific pytorch model from some directory
        '''
        return torch.load(os.path.join(path))