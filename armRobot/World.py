from framework.World import BaseWorld
from armRobot.RobotArm import RobotArm
import numpy as np
import torch



class World (BaseWorld):
    def __init__(self, arm_configuration, dynamics_params_no = -1,  dynamics_reset_min = 0.3, dynamics_reset_range = 0.4,
                 start_config_std = 0.01, target_pos = None, action_resolution = 0.1, change_target_pos = True,
                 change_start_config = True, success_radius = 0.015, max_number_of_steps = 200):
        '''
        World class for the armRobot game.
        :param arm_configuration: [List of numpy arrays of size 2]. List of initial effector positions for the arm,
        which defines the arm configuration. Each position (x,y) must be such that 0<x<1, 0<y<1
        :param dynamics_params_no: [int] number of dynamics parameters for the robot arm, must be at least as large as the
        arms configurations.
        :param dynamics_reset_min: [float between 0 and 1] Minimum value for the dynamics parameters, used when setting and resetting the
        dynamcis.
        :param dynamics_reset_range: [float between 0 and 1] The dynamics can take values within dynamics_reset_min : dynamics_reset_min + dynamics_reset_range
        :param start_config_std [float between 0 and 1]: When the start configuration, the joint angles will vary randomly by with standard deviation
        start_config_std.
        :param target_pos: [array of length 2]. Each dimention has to be between 0 and 1.
         If given, the initial target position is set to this value.
        :param action_resolution: [float between 0 and 1] order of magnitude of each action, where an action_resolution of 1 means the ground truth
        actions reach the target in 1 sep
        :param change_target_pos: [bool] Whether to change the target position upon each reset
        :param change_start_config: [bool] whether to change the start configuration upon each reset
        :param success_radius:[float between 0 and 1] The radius around the target position where the reaching is considered a success.
        :param max_number_of_steps: [int] Maximum number of allowed steps per episode before terminating with failure.
        '''
        super(World,self).__init__(success_radius,max_number_of_steps, change_target_pos,change_start_config)

        # Instantiate robot arm
        self.robot_arm = RobotArm(arm_configuration, dynamics_params_no, dynamics_reset_min, dynamics_reset_range)


        self.start_config_std = start_config_std

        if (target_pos is None):
            random_angle = 2 * np.pi * np.random.rand()
            max_len = np.sum(self.robot_arm.arm_lengths)
            random_radius_squared = np.random.uniform(high=max_len * max_len)
            random_radius = np.sqrt(random_radius_squared)
            self.target_pos = np.array([random_radius * np.cos(random_angle),
                                        random_radius * np.sin(random_angle)]) + self.robot_arm.origin_position
        else:
            self.target_pos = target_pos
            if (not self.robot_arm.reach_check(target_pos)):
                raise ValueError('Cannot Reach the goal!')

        # Set the state and record initial configurations
        self.__state = np.array((*self.robot_arm.effector_positions_vector(), *self.robot_arm.joint_angles(),
                                 *self.target_pos))

        self.initial_state = np.copy(self.__state)
        self.initial_arm_configuration = self.robot_arm.effector_positions

        # Set the action resolution (so that the ground truth does not reach the goal in one step)
        self.action_resolution = action_resolution

        # Standardisation constants
        self.__number_of_angles = len(self.robot_arm.joint_angles())
        self.__number_of_arms = len(self.robot_arm.arm_lengths)

        self.__effector_pos_half_range = 0.5 + np.sum(self.robot_arm.arm_lengths) / 2.
        self.__effector_pos_max_deviation = np.sum(self.robot_arm.arm_lengths) / 2.

        self.__angles_half_range = 0
        self.__angles_max_deviation = np.pi

        self.__pos_half_range = 0.5
        self.__pos_max_deviation = 0.5

        self.__effective_dynamics_min = self.robot_arm.effective_dynamics_min()
        self.__effective_dynamics_ave = (self.robot_arm.effective_dynamics_min()+self.robot_arm.effective_dynamics_max())/2

    @property
    def state(self):
        return np.copy(self.__state)

    @state.setter
    def state(self, val):
        self.__state = val

    @property
    def state_size(self):
        return len(self.__state)

    @property
    def action_size(self):
        return self.robot_arm.number_of_angles
    @property
    def dynamics(self):
        return self.robot_arm.dynamics_params

    @dynamics.setter
    def dynamics(self, val):
        self.robot_arm.set_dynamics(val)
        
    def update_target_pos(self, new_pos):
        self.target_pos = np.array(new_pos)
        self.__state[-2:] = self.target_pos
        
    def optimal_policy(self, state, dynamics = None):
        '''
        Return the optimal policy to reach the goal from the given state.
        :param state: The state from which to calculate the optimal policy.
        :param dynamics: If given, calculate the optimal policy w.r.t these dynamics.
        '''
        target = state[-2:]
        effector_state_dim = len(self.robot_arm.effector_positions_vector())
        effector_pos = state[:effector_state_dim]
        effector_pos = np.split(effector_pos,len(effector_pos)/2)[::-1]

        target_positions = self.robot_arm.fabrik(target,external_positions = np.copy(effector_pos))
        action = self.robot_arm.get_angle_actions(target_positions, initial_positions = effector_pos,
                                                  iterations = 1/self.action_resolution, correct_for_dynamics = True,
                                                  external_dynamics = dynamics)

        return action

    def reward(self, state):
        '''Return the reward, which is the negative of the Euclidean distance between the end effector and the target position.'''
        pos = state[:2]

        dist = np.linalg.norm(self.target_pos - pos)
        return -dist

    def action_response(self, action):
        '''Apply the action to the environment, and return the new state, the reward and whether the episode terminated.
        The episode terminates if the max number of steps is hit, or if the end effector is within the success radius
        of the target position.'''

        # Add the (state,action) pair to the current episode trajectory
        self.trajectory.append(np.concatenate([self.__state, action], axis=0))

        # Move the arm
        self.robot_arm.move_joints(action)

        # Calculate the new state, the reward and update the number of steps in the current episode
        new_state = np.array((*self.robot_arm.effector_positions_vector(), *self.robot_arm.joint_angles(), *self.target_pos))
        reward = self.reward(new_state)
        self.steps += 1

        # Determine whether it is the end of the episode and whether the robot reached the target
        if (np.linalg.norm((new_state[:2] - self.target_pos)) <= (self.success_radius)):
            self.terminated = True
            self.success = True

        if (self.steps >self.max_number_of_steps):
            self.terminated = True

        self.__state = new_state


        return new_state, reward, self.terminated

    def simulate_next_state(self,state,action,dynamics, batch = False):
        if(batch):
            effector_state_dim = len(self.robot_arm.effector_positions_vector())
            initial_angles = state[:,effector_state_dim:effector_state_dim + self.action_size]
            effective_dyns = self.robot_arm.get_dyns_from_params(dynamics, batch)
            next_effector_positions, new_angles = self.robot_arm.apply_angle_changes(action, initial_angles,
                                                                                     effective_dyns, batch)
            next_state = np.concatenate([np.concatenate(next_effector_positions[::-1], axis=1),
                                   np.concatenate(new_angles,axis =1),
                                   state[:,-2:]], axis = 1)  # *self.robot_arm.joint_angles(positions=next_effector_positions)

            return next_state

        else:
            effector_state_dim = len(self.robot_arm.effector_positions_vector())
            initial_angles = state[effector_state_dim:effector_state_dim + self.action_size]
            effective_dyns = self.robot_arm.get_dyns_from_params(dynamics)
            next_effector_positions, new_angles = self.robot_arm.apply_angle_changes(action, initial_angles,
                                                                                     effective_dyns)
            next_state = np.array([*np.concatenate(next_effector_positions[::-1]),
                                   *new_angles,
                                   *state[-2:]])  # *self.robot_arm.joint_angles(positions=next_effector_positions)

            return next_state


    def reset(self, reset_dynamics = False, from_state = None, specific_dynamics = None):
        '''
        Reset the state of the world at the end of an episode.
        :param reset_dynamics: If true, reset the dynamics of the system according to the relevant parameters.
        :param from_state: If present, set the initial state to the world to this state.
        :param specific_dynamics: If present, set the dynamics parameters of the systems to those.
        '''


        if (from_state is None):
            if(self.change_start_config):
                # Set the start configuration to the initial configuration plus some minor perturbation
                self.robot_arm.effector_positions = self.initial_arm_configuration
                angles = self.robot_arm.joint_angles(self.initial_arm_configuration)
                angles +=  self.start_config_std*np.random.randn(len(self.initial_arm_configuration)-1)
                self.robot_arm.effector_positions = self.robot_arm.effector_pos_from_angles(angles)
            else:
                self.robot_arm.effector_positions = self.initial_arm_configuration

            if (self.change_target_pos):
                random_angle = 2 * np.pi * np.random.rand()
                max_len = np.sum(self.robot_arm.arm_lengths)
                random_radius_squared = np.random.uniform(high=max_len * max_len)
                random_radius = np.sqrt(random_radius_squared)
                self.target_pos = np.array([random_radius * np.cos(random_angle), random_radius * np.sin(random_angle)])+self.robot_arm.origin_position

            # Check that there are no inconsistencies in automatic reset mode
            if (not self.robot_arm.reach_check((self.target_pos))):
                raise ValueError('Cannot reach the goal!')

        else:
            # Don't allow reset to unreachable states
            if (not self.robot_arm.reach_check(from_state[-2:])):
                print('Cannot reach the goal!')
                return

            #If a state is given, reset to that state
            effector_state_dim = len(self.robot_arm.effector_positions_vector())
            pos_to_reset = self.robot_arm.effector_pos_from_angles(from_state[effector_state_dim:self.robot_arm.number_of_angles + effector_state_dim])  # bug danger
            self.robot_arm.effector_positions = pos_to_reset
            self.target_pos = from_state[-2:]


        self.robot_arm.check_consistency()


        if (specific_dynamics is not None):
            self.robot_arm.set_dynamics(specific_dynamics)
        elif (reset_dynamics):
            self.robot_arm.reset_dynamics()

        # Reset the trajectory tracking parameters
        self.trajectory = []
        self.__state = np.array((*self.robot_arm.effector_positions_vector(), *self.robot_arm.joint_angles(), *self.target_pos))
        self.initial_state = np.copy(self.__state)
        self.steps = 0
        self.terminated = False
        self.success = False


    def standardise(self, object, type, batch = False):
        '''
        Map each dimention of the object input to a standard range.
        :param object: [torch tensor or numpy array] The object to standardise.
        :param type: [string] The type of the object 'state', 'action' or 'dynamics.
        '''
        if(type == 'state'):
            return self.standardise_state(object, batch)
        elif(type == 'action'):
            return self.standardise_action(object)
        elif(type == 'dynamics'):
            return self.standardise_dynamics(object)
        else:
            raise ValueError('Can only standardise states,  actions or dynamics')


    def un_standardise(self, object, type, batch = False):
        '''
        Reverse the effect of standardising.
        :param object: [torch tensor or numpy array] The object to un-standardise.
        :param type: [string] The type of the object 'state', 'action' or 'dynamics'.
        '''
        if(type == 'state'):
            return self.un_standardise_state(object, batch)
        elif(type == 'action'):
            return self.un_standardise_action(object)
        elif(type =='dynamics'):
            return self.un_standardise_dynamics(object)

        else:
            raise ValueError('Can only un-standardise states, actions or dynamics')


    def standardise_state(self,state, batch = False):
        '''Map a state to a standard range'''
        if(batch):
            effector_state_dim = len(self.robot_arm.effector_positions_vector())
            effector_pos = state[:,:effector_state_dim]
            joint_angles = state[:,effector_state_dim:effector_state_dim+self.__number_of_angles]
            target = state[:, -2:]
            cat_dim = 1
        else:
            effector_state_dim = len(self.robot_arm.effector_positions_vector())
            effector_pos = state[:effector_state_dim]
            joint_angles = state[effector_state_dim:effector_state_dim + self.__number_of_angles]
            target = state[-2:]
            cat_dim=0

        if (isinstance(state, torch.Tensor)):
            effector_pos_new = torch.div(torch.add(effector_pos, -self.__effector_pos_half_range),self.__effector_pos_max_deviation)
            joint_angles_new = torch.div(torch.add(joint_angles, -self.__angles_half_range),self.__angles_max_deviation)
            target_new =  torch.div(torch.add(target, -self.__pos_half_range),-self.__pos_max_deviation)
            return torch.cat([effector_pos_new,joint_angles_new,target_new],dim=cat_dim)
        elif (isinstance(state, np.ndarray)):
            effector_pos_new =  np.divide((effector_pos - self.__effector_pos_half_range),self.__effector_pos_max_deviation)
            joint_angles_new =  np.divide((joint_angles - self.__angles_half_range),self.__angles_max_deviation)
            target_new = np.divide((target - self.__pos_half_range),self.__pos_max_deviation)
            return np.concatenate([effector_pos_new,joint_angles_new,target_new],axis=cat_dim)
        else:
            raise ValueError('Needs to be either np array or tensor')



    def standardise_action(self,action):
        '''Map an action to a standard range'''
        action_order = (self.__angles_max_deviation * self.action_resolution) / self.__effective_dynamics_ave
        if (isinstance(action, torch.Tensor)):
            return torch.div(action, action_order)
        elif (isinstance(action, np.ndarray)):
            return np.divide(action, action_order)
        else:
            raise ValueError('Needs to be either np array or tensor')


    def standardise_dynamics(self,dynamics):
        '''Map a dynamics vector to a standard range'''

        if(np.isclose(self.robot_arm.dynamics_reset_range,0.)):
            return dynamics

        if (isinstance(dynamics, torch.Tensor)):
            return torch.div(
                torch.add(dynamics, -(self.robot_arm.dynamics_reset_min + (self.robot_arm.dynamics_reset_range / 2.))),
                (self.robot_arm.dynamics_reset_range / 2.))
        elif (isinstance(dynamics, np.ndarray)):
            return np.divide((dynamics - (self.robot_arm.dynamics_reset_min + (self.robot_arm.dynamics_reset_range / 2.))),
                             (self.robot_arm.dynamics_reset_range / 2.))
        else:
            raise ValueError('Needs to be either np array or tensor')


    def un_standardise_state(self,state, batch = False):
        '''Map a state from its standardised form back to its original space'''
        if(batch):
            effector_state_dim = len(self.robot_arm.effector_positions_vector())
            effector_pos = state[:,:effector_state_dim]
            joint_angles = state[:,effector_state_dim:effector_state_dim + self.__number_of_angles]

            target = state[:,-2:]
            cat_dim = 1
        else:
            effector_state_dim = len(self.robot_arm.effector_positions_vector())
            effector_pos = state[:effector_state_dim]
            joint_angles = state[effector_state_dim:effector_state_dim + self.__number_of_angles]

            target = state[-2:]
            cat_dim=0

        if (isinstance(state, torch.Tensor)):
            effector_pos_new =  torch.add(torch.mul(effector_pos,self.__effector_pos_max_deviation), self.__effector_pos_half_range)
            joint_angles_new = torch.add(torch.mul(joint_angles,self.__angles_max_deviation), self.__angles_half_range)
            target_new =  torch.add(torch.mul(target,self.__pos_max_deviation), self.__pos_half_range)
            return torch.cat([effector_pos_new,joint_angles_new,target_new],dim =cat_dim)
        elif (isinstance(state, np.ndarray)):
            effector_pos_new = np.multiply(effector_pos, self.__effector_pos_max_deviation) + self.__effector_pos_half_range
            joint_angles_new = np.multiply(joint_angles, self.__angles_max_deviation) + self.__angles_half_range
            target_new = np.multiply(target, self.__pos_max_deviation) + self.__pos_half_range
            return np.concatenate([effector_pos_new,joint_angles_new,target_new],axis=cat_dim)
        else:
            raise ValueError('Needs to be either np array or tensor')


    def un_standardise_action(self,action):
        '''Map an action from its standardised form back to its original space'''
        action_order = (self.__angles_max_deviation * self.action_resolution )/ self.__effective_dynamics_ave
        if (isinstance(action, torch.Tensor)):
            return torch.mul(action, action_order)
        elif (isinstance(action, np.ndarray)):
            return np.multiply(action, action_order)
        else:
            raise ValueError('Needs to be either np array or tensor')


    def un_standardise_dynamics(self,dynamics):
        '''Map a dynamics vector from its standardised form back to its original space'''

        if (np.isclose(self.robot_arm.dynamics_reset_range, 0.)):
            return dynamics

        if (isinstance(dynamics, torch.Tensor)):
            return torch.add(torch.mul(dynamics, (self.robot_arm.dynamics_reset_range / 2.)),
                             (self.robot_arm.dynamics_reset_min + (self.robot_arm.dynamics_reset_range / 2.)))
        elif (isinstance(dynamics, np.ndarray)):
            return np.multiply(dynamics, (self.robot_arm.dynamics_reset_range / 2.)) + (
                        self.robot_arm.dynamics_reset_min + (self.robot_arm.dynamics_reset_range / 2.))
        else:
            raise ValueError('Needs to be either np array or  tensor')


if __name__=='__main__':

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    world_params = {'arm_configuration': [np.array([0.5, 0.5]), np.array([0.5, 0.55]), np.array([0.5, 0.6]),
                                          np.array([0.5, 0.65]), np.array([0.5, 0.7]), np.array([0.5, 0.75]),
                                          np.array([0.5, 0.8]), np.array([0.5, 0.85]), np.array([0.5, 0.9])],
                    'dynamics_params_no': 20,  # DYNAMICS
                    'dynamics_reset_min': 0.2,
                    'dynamics_reset_range': 0.6,
                    'start_config_std': 0.01,
                    'action_resolution': 0.1,
                    'target_pos': np.array([0.75, 0.25]),
                    'change_target_pos': True,
                    'change_start_config': True,
                    'success_radius': 0.02,
                    'max_number_of_steps': 65}

    w = World(**world_params)
    action = w.optimal_policy(w.state)



    torch_state = torch.from_numpy(w.state)
    torch_action = torch.from_numpy(action)
    torch_dynamcis = torch.from_numpy(w.dynamics)

    sim_s = w.simulate_next_state(torch_state, torch_action,torch_dynamcis)


    new_s ,_,_ = w.action_response(action)

    print(sim_s)
    print(new_s)
    print(np.isclose(sim_s,new_s).all())
    assert(np.isclose(sim_s,new_s).all())

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    w = World(**world_params)
    action = w.optimal_policy(w.state)

    states = np.concatenate([w.state.reshape((1,-1))]*16,axis = 0)
    actions = np.concatenate([action.reshape((1,-1))]*16, axis = 0)
    dynamics = np.concatenate([w.dynamics.reshape(1,-1)]*16, axis = 0)

    # torch_states = torch.from_numpy(states)
    # torch_actions = torch.from_numpy(actions)
    # torch_dynamics = torch.from_numpy(dynamics)




    sim_s = w.simulate_next_state(states, actions,dynamics,True)


    new_s ,_,_ = w.action_response(action)

    new_s = np.concatenate([new_s.reshape(1,-1)]*16,axis=0)

    print(sim_s)
    print(new_s)
    print(np.isclose(sim_s,new_s).all())
    assert(np.isclose(sim_s,new_s).all())