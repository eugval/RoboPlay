import numpy as np
import pickle
from abc import ABC
from copy import deepcopy



class DataBuffer(ABC):
    def __init__(self):
        self.__states = []
        self.__actions = []
        self.__correct_actions = []
        self.__dynamics = []
        self.__traj_idx = []
        self.__initial_configs = []

        self.__test_states = []
        self.__test_actions = []
        self.__test_correct_actions = []
        self.__test_dynamics = []
        self.__test_traj_idx = []
        self.__test_initial_configs = []


    @property
    def states(self):
        return self.__states.copy()

    @property
    def test_states(self):
        return self.__test_states.copy()


    @property
    def actions_taken(self):
        return self.__actions.copy()


    @property
    def correct_actions(self):
        return self.__correct_actions.copy()

    @property
    def test_correct_actions(self):
        return self.__test_correct_actions.copy()

    @property
    def trajectory_indices(self):
        return self.__traj_idx.copy()

    @property
    def test_trajectory_indices(self):
        return self.__test_traj_idx.copy()

    @property
    def dynamics(self):
        return self.__dynamics.copy()

    @property
    def test_dynamics(self):
        return self.__test_dynamics.copy()

    @property
    def initial_configs(self):
        return self.__initial_configs.copy()

    @property
    def test_initial_configs(self):
        return self.__test_initial_configs.copy()

    @test_initial_configs.setter
    def test_initial_configs(self,val):
        self.__test_intitial_configs = val

    @property
    def number_of_trajectories(self):
        return len(self.__initial_configs)

    @property
    def number_of_test_trajectories(self):
        return len(self.__test_initial_configs)

    def states_len(self, test = False):
        if(test):
            return len(self.__test_states)
        else:
            return len(self.__states)

    def actions_len(self,test = False):
        if(test):
            return len(self.__test_actions)
        else:
            return len(self.__actions)

    def state_size(self, test = False):
        if(self.states_len(test)>0):
            return len(self.__states[0])
        else:
            raise ValueError('No states in the dataset')

    def action_size(self, test = False):
        if(self.actions_len(test)>0):
            return len(self.__actions[0])
        else:
            raise ValueError('No actions in the dataset')


    def get_state(self,idx, test = False):
        if(test):
            return np.copy(self.__test_states[idx])
        else:
            return np.copy(self.__states[idx])

    def get_action_taken(self,idx,test = False):
        if(test):
            return np.copy(self.__test_actions[idx])
        else:
            return np.copy(self.__actions[idx])

    def get_trajectory_index(self,idx,test = False):
        if(test):
            return np.copy(self.__test_traj_idx[idx])
        else:
            return np.copy(self.__traj_idx[idx])

    def get_initial_config(self,idx, test = False):
        if(test):
            return deepcopy(self.__test_initial_configs[idx])
        else:
            return deepcopy(self.__initial_configs[idx])

    def get_dynamics_params(self,idx,test = False):
        if(test):
            return np.copy(self.__test_dynamics[idx])
        else:
            return np.copy(self.__dynamics[idx])

    def get_optimal_action(self,idx,test=False):
        if(test):
            return np.copy(self.__test_correct_actions[idx])
        else:
            return np.copy(self.__correct_actions[idx])

    def add_state(self, state, test=False):
        if(test):
            self.__test_states.append(state)
        else:
            self.__states.append(state)

    def add_action_taken(self, action, test = False):
        if(test):
            self.__test_actions.append(action)
        else:
            self.__actions.append(action)

    def add_correct_action(self, correct_action, test=False):
        if(test):
            self.__test_correct_actions.append(correct_action)
        else:
            self.__correct_actions.append(correct_action)

    def add_dynamics(self, dynamics, test=False):
        if(test):
            self.__test_dynamics.append(dynamics)
        else:
            self.__dynamics.append(dynamics)

    def add_trajectory_index(self, trajectory_index, test=False):
        if(test):
            self.__test_traj_idx.append(trajectory_index)
        else:
            self.__traj_idx.append(trajectory_index)

    def add_initial_config(self, initial_config, test=False):
        if(test):
            self.__test_initial_configs.append(initial_config)
        else:
            self.__initial_configs.append(initial_config)

    def get_full_data_dict(self):
        return {'states': self.__states,
                'actions_taken': self.__actions,
                'correct_actions': self.__correct_actions,
                'dynamics': self.__dynamics,
                'traj_idx': self.__traj_idx,
                'initial_configs': self.__initial_configs,
                'test_states': self.__test_states,
                'test_actions':self.__test_actions,
                'test_correct_actions': self.__test_correct_actions,
                'test_dynamics': self.__test_dynamics,
                'test_traj_idx': self.__test_traj_idx,
                'test_initial_configs': self.__test_initial_configs}

    def save(self, file):
        data = {'states': self.__states,
                'actions_taken': self.__actions,
                'correct_actions': self.__correct_actions,
                'dynamics': self.__dynamics,
                'traj_idx': self.__traj_idx,
                'initial_configs': self.__initial_configs,
                'test_states': self.__test_states,
                'test_actions':self.__test_actions,
                'test_correct_actions': self.__test_correct_actions,
                'test_dynamics': self.__test_dynamics,
                'test_traj_idx': self.__test_traj_idx,
                'test_initial_configs': self.__test_initial_configs}

        pickle.dump(data, open(file, 'wb'))

    def load(self, load_from):
        if(isinstance(load_from,str)):
            data = pickle.load(open(load_from, 'rb'))
            self.__states = data['states']
            self.__actions = data['actions_taken']
            self.__correct_actions = data['correct_actions']
            self.__dynamics = data['dynamics']
            self.__traj_idx = data['traj_idx']
            self.__initial_configs = data['initial_configs']
            self.__test_states = data['test_states']
            self.__test_actions = data['test_actions']
            self.__test_correct_actions = data['test_correct_actions']
            self.__test_dynamics = data['test_dynamics']
            self.__test_traj_idx = data['test_traj_idx']
            self.__test_initial_configs = data['test_initial_configs']
        elif(isinstance(load_from,dict)):
            data = load_from
            self.__states = data['states']
            self.__actions = data['actions_taken']
            self.__correct_actions = data['correct_actions']
            self.__dynamics = data['dynamics']
            self.__traj_idx = data['traj_idx']
            self.__initial_configs = data['initial_configs']
            self.__test_states = data['test_states']
            self.__test_actions = data['test_actions']
            self.__test_correct_actions = data['test_correct_actions']
            self.__test_dynamics = data['test_dynamics']
            self.__test_traj_idx = data['test_traj_idx']
            self.__test_initial_configs = data['test_initial_configs']


    def reset(self, reset_test_configs=False):
        self.__states = []
        self.__actions = []
        self.__correct_actions = []
        self.__dynamics = []
        self.__traj_idx =[]
        self.__initial_configs = []

        if(reset_test_configs):
            self.reset_test()

    def reset_test(self):
        self.__test_states = []
        self.__test_actions = []
        self.__test_correct_actions = []
        self.__test_dynamics = []
        self.__test_traj_idx = []
        self.__test_initial_configs = []

    def generate_initial_configs(self,world, number_of_configs,reset_dynamics=True):
        '''
        Return a list of [(state,dynamics)] initial configurations by repeatedly resetting the world.
        :param world: The world environment generating the initial configurations.
        :param number_of_configs: [int] The number of initial configurations to generate.
        :param reset_dynamics: [bool] Whether to reset the dynamics as well.
        :return: [list of (state,dynamics)] tuples containing initial configurations for the given world.
        '''
        print('Generating initial configurations...')
        initial_configs = []

        for i in range(number_of_configs):
            world.reset(reset_dynamics=reset_dynamics)
            initial_configs.append((world.state,world.dynamics))

        return initial_configs


    def expand_inputs_for_training(self,state, idx,  world, standardise,
                                   number_of_input_states,final_size, test=False):
        '''
        Expand a state to a state-action sequence that is consistent with its place in the trajectory. If at the beginning
        of the trajectory, pad with zeros. State is expanded backwards.
        :param state: [np array] The state to expand.
        :param idx: [int] The index of the state to expand.
        :param world: The world.
        :param standardise: [bool] Whether to standardise the states action sequences
        :param number_of_input_states: [int] how many states the final input needs to have.
        :param input_size: [int] The size of the final output.
        :return: The expanded state
        '''

        if (standardise):
            state = world.standardise(state, 'state')

        count = number_of_input_states - 1
        trajectory_index = self.get_trajectory_index(idx,test)

        while (count > 0 and trajectory_index > 0):
            idx -= 1
            sample_state = self.get_state(idx,test)
            sample_action = self.get_action_taken(idx,test)

            if (standardise):
                sample_state = world.standardise(sample_state, 'state')
                sample_action = world.standardise(sample_action, 'action')

            state = np.concatenate([sample_state, sample_action, state], axis=0)
            count -= 1
            trajectory_index -= 1


        if (state.size < final_size):
            zeros_array = np.zeros(final_size)
            zeros_array[-state.size:] = state
            state = zeros_array

        return state

    def generate_dataset(self, world, number_of_trajectories, dynamics_reset_interval=1, noise_injection=0,
                         state_reset_interval=1, test = False):
        '''
        Generate a dataset from scratch by running the optimal policy, potentially augmented with noise-injected states.
        :param world: The world instance.
        :param number_of_trajectories: [int] The number of trajectory rollouts to perform.
        :param dynamics_reset_interval: [int] How often to reset the dynamics of the world when rolling out trajectories.
        :param noise_injection:  [float between 0 and 1] When rolling out the trajectory, inject noise to the optimal action
        with a standard deviation given by this parameter in order to get the state distribution. (similar to DART).
        :param state_reset_interval: [int] How often to change the initial state of the world when rolling out trajectories.
        :param test: [boo] If true, populates the test set
        '''
        print('Generating dataset...')
        if (state_reset_interval < 1):
            state_reset_interval == 1

        if (dynamics_reset_interval < 1):
            dynamics_reset_interval = number_of_trajectories + 1


        while (number_of_trajectories > 0):
            i = 0
            dyn = world.dynamics
            self.add_initial_config((world.state, dyn),test)

            while (not world.terminated):
                state = world.state
                optimal_action = world.optimal_policy(state)

                if (noise_injection > 0):
                    actual_action = np.random.normal(optimal_action, noise_injection, len(optimal_action))
                else:
                    actual_action = optimal_action

                self.add_state(state,test)
                self.add_action_taken(actual_action,test)
                self.add_correct_action(optimal_action,test)
                self.add_dynamics(dyn,test)
                self.add_trajectory_index(i,test)



                world.action_response(actual_action)
                i += 1

            if (number_of_trajectories % state_reset_interval == 0):
                from_state = None
            else:
                from_state = world.initial_state

            if (number_of_trajectories % dynamics_reset_interval == 0):
                world.reset(reset_dynamics=True, from_state=from_state)
            else:
                world.reset(from_state=from_state)

            number_of_trajectories -= 1


    def add_policy_generated_data(self, world, states, actions, dynamics, trajectory_indices):
        '''
        Add data generated by some policy to the data buffer. Used to add DAGGER generated data.
        :param world: The world instance.
        :param states: [list of states] The states sequence generated by rolling out the external policy.
        :param actions: [list of actions] The actions sequence taken by the external policy.
        :param dynamics: [list of dynamics] The dynamics experienced during each of the above states/actions
        :param trajectory_indices: [list of ints] List of indices representing the timestep in the trajectory for each
        the states/actions/dynamics in the lists above.
        :return:
        '''
        print('Adding policy generated data...')

        for i, state in enumerate(states):
            dyns = dynamics[i]
            traj_idx = trajectory_indices[i]
            action_taken = actions[i]
            optimal_action = world.optimal_policy(state, dynamics=dyns)

            self.add_state(state)
            self.add_action_taken(action_taken)
            self.add_correct_action(optimal_action)
            self.add_dynamics(dyns)
            self.add_trajectory_index(traj_idx)

            # Only add to the initial configurations when the trajectory index is 0
            if (traj_idx == 0):
                self.add_initial_config((state, dyns))

