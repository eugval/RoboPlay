import numpy as np
import collections


class RobotArm(object):
    def __init__(self, initial_effector_positions, dynamics_params_no=-1, dynamics_reset_min=0.3,
                 dynamics_reset_range=0.4):
        '''
         Class for a 2D robot arm, with determined dynamics and an arbitrary number of joints.
        :param initial_effector_positions: [list of np arrays of size 2] The list of the effector coordinates of the arm,
        define the configuration of the arm.
        :param dynamics_params_no: [int] Number of dynamics parameters accepted by the arm. If negative, the arm has no
        dynamics.
        :param dynamics_reset_min: [float between 0 and 1] Upon reset, this is the minimum value the dynamics parameters
        of the arm can take
        :param dynamics_reset_range: [float between 0 and 1] The maximum value for the dynamics parameters of the arm
        are dynamics_reset_min + dynamics_reset_range.
        '''
        # The full state of the arm is encoded in the positions of the effectors
        self.__effector_positions = initial_effector_positions

        self.check_consistency()


        # Dynamics reset parameters
        self.dynamics_params_no = dynamics_params_no
        self.dynamics_reset_min = dynamics_reset_min
        self.dynamics_reset_range = dynamics_reset_range

        # Add dynamics
        if (dynamics_params_no == -1):
            self.dynamics_params_no = len(initial_effector_positions)
            self.dynamics_reset_min = 0.5
            self.dynamics_reset_range = 0
            self.reset_dynamics()
        else:
            self.reset_dynamics()

    @property
    def number_of_effectors(self):
        '''
        Return the number of effector positions in the arm (number of joints + end effector).
        '''
        return len(self.__effector_positions)

    @property
    def number_of_angles(self):
        '''
        Return the number of angles (or joints) for control.
        '''
        return len(self.__effector_positions) - 1

    @property
    def effector_positions(self):
        '''Return the list of effector positions [(xi,yi)].'''
        return self.__effector_positions.copy()

    @effector_positions.setter
    def effector_positions(self, val):
        '''Set the effector positions, check for consistency.'''
        self.__effector_positions = val
        self.check_consistency()

    @property
    def origin_position(self):
        '''Return the position of the first effector (origin of the arm).'''
        return np.copy(self.effector_positions[0])

    @property
    def end_effector_position(self):
        '''Return the position of the end effector.'''
        return np.copy(self.effector_positions[-1])

    @property
    def arm_lengths(self):
        '''Return the arm link lenghts'''
        return self.get_arm_lengths()

    @property
    def dynamics_params(self):
        '''Return the dynamics parameters controlling the dynamics of the arm.'''
        return np.copy(self.__dynamics_params)

    @property
    def effective_dynamics(self):
        '''Return the dynamics for each joint angle (multiplicative bias).'''
        return np.copy(self.__effective_dynamics)

    def effector_positions_vector(self, reverse=True):
        '''Return a numpy array of the concatenated effector positions.'''
        if (reverse):
            # Enf-effector first
            return np.concatenate(self.__effector_positions[::-1])
        else:
            # Origin first
            return np.concatenate(self.__effector_positions)

    def set_dynamics(self, params):
        '''Set the dynamics (both effective and parameters) given a new set of parameters.'''
        self.__dynamics_params = params
        self.__effective_dynamics = self.get_dyns_from_params(params)

    def reset_dynamics(self):
        '''Reset the dynamics according to the dynamics reset metadata.'''
        # Check there is enough dynamics parameters to get the effective dynamics
        assert len(self.__effector_positions) <= self.dynamics_params_no

        dyn_params = []
        for i in range(self.dynamics_params_no):
            dyn_params.append(self.dynamics_reset_range * np.random.rand(1)[0] + self.dynamics_reset_min)

        self.set_dynamics(np.array(dyn_params))

    def get_dyns_from_params(self, dynamics_params, batch=False):
        ''' Calculate the effective dynamics using the dynamics parameters in some arbitrary way.
            If batch= True, calculated it for a batch of dynamics parameters, where dim = 0 is the batch dimention
        '''
        if(batch):
            dyn = []
            i = 0
            while len(dyn) < len(self.__effector_positions) - 1:
                dyn.append((((dynamics_params[:, i] + dynamics_params[:,i + 1]) * dynamics_params[:,i]) / dynamics_params[:,-1 - i]).reshape((-1,1)))
                i+=1
            return np.concatenate(dyn, axis=1)
        else:
            dyn = []
            i = 0
            while len(dyn) < len(self.__effector_positions) - 1:
                dyn.append(
                    ((dynamics_params[i] + dynamics_params[i + 1]) * dynamics_params[i]) / dynamics_params[-1 - i])
                i += 1
            return np.array(dyn)


    def effective_dynamics_min(self):
        ''' Return the minimum possible value of the effective dynamics given the parameter reset metadata.'''
        return (2 * self.dynamics_reset_min * self.dynamics_reset_min) / (
                    self.dynamics_reset_min + self.dynamics_reset_range)

    def effective_dynamics_max(self):
        ''' Return the minimum possible value of the effective dynamics given the parameter reset metadata.'''
        dyns_reset_max = self.dynamics_reset_min+self.dynamics_reset_range
        return (2 * dyns_reset_max * dyns_reset_max) / (self.dynamics_reset_min)

    def check_consistency(self):
        '''Check that the robot effectors are in the world and that they can reach everywhere within the arm's circle of reach.'''

        # All positions must be between 0 and 1
        for pos in self.effector_positions:
            if( pos[0]<0 or pos[0]>1 or pos[1]<0 or pos[1]>1):
                raise ValueError('The robot effector positions are out of range')

        # The length of the largest arm needs to be smaller than the sum of the rest of the arms
        arm_lengths = self.arm_lengths
        max_idx = np.argmax(arm_lengths)
        max_len = arm_lengths[max_idx]
        sum_lens = np.sum(arm_lengths[np.arange(len(arm_lengths)) != max_idx])
        if ((not np.isclose(max_len, sum_lens) and (max_len > sum_lens))):
            raise ValueError('The robot arm cannot reach everywhere in its disk')

    def link_vectors(self, positions=None):
        '''
        Return a list of arm link vectors.
        :param positions: If effector positions are passed as an argument, return the link vectors of this list instead.
        '''
        link_vectors = []

        if (positions is not None):
            j_p = positions
        else:
            j_p = self.effector_positions

        for i in range(len(self.effector_positions) - 1):
            # The link vectors are the difference in the positions of the effectors
            link_vectors.append(j_p[i + 1] - j_p[i])

        return link_vectors

    def get_arm_lengths(self, external_positions=None):
        '''
        Return the arm link lenghts.
        :param external_positions: If effector positions are passed as an argument, return the arm lenghts of this list instead.
        '''
        arm_lengths = []

        if (external_positions is None):
            j_pos = self.effector_positions
        else:
            j_pos = external_positions

        for i in range(len(j_pos) - 1):
            # The norm of the vector between the two effectors in cartesian coordinates
            arm_lengths.append(np.linalg.norm(j_pos[i + 1] - j_pos[i]))
        return np.array(arm_lengths)


    def joint_angles(self, positions=None):
        '''
        Retun the joint angles of the current robot configuration.
        :param positions: If a list of effector positions is passed as an argument, return the joint angles of this list instead.
        '''
        if (positions is not None):
            links = self.link_vectors(positions)
        else:
            links = self.link_vectors()
        joint_angles = []
        original_frame_angles = []

        # The first joint angle is the signed arctan of the x,y coordinates of the first link vector
        first_angle = np.arctan2(links[0][1], links[0][0])
        joint_angles.append(first_angle)
        original_frame_angles.append(first_angle)

        for i in range(1, len(links)):
            # Each joint angle is the angle with the x-axis - the previous angle with the x-axis
            frame_angle = np.arctan2(links[i][1], links[i][0])
            angle = frame_angle - original_frame_angles[-1]
            original_frame_angles.append(frame_angle)
            joint_angles.append(angle)

        return np.array(joint_angles)

    def effector_pos_from_angles(self, joint_angles, batch= False):
        '''
        Infer the effector positions required to produced the given joint angles by forward kinematics.
        :param batch: Do it batch mode, where dim = 0 is the batch dimention
        :param joint_angles: Joint angles, list or array of lengths the number of effectors-1.
        '''

        if(batch):
            # Check that the list of joint_angles has the right length
            assert len(joint_angles) == len(self.effector_positions) - 1
            arm_lengths = self.arm_lengths

            effector_pos = [np.zeros((joint_angles[0].size,2))]
            cumulative_angle = np.zeros((joint_angles[0].size,1))
            for i in range(len(arm_lengths)):
                # Working out the forward kinematics, placing the first link at the origin
                angle = joint_angles[i] + cumulative_angle
                cumulative_angle += joint_angles[i]
                calc_pos = effector_pos[-1] + arm_lengths[i] * np.concatenate([np.cos(angle), np.sin(angle)],axis=1)
                effector_pos.append(calc_pos)

            # Adding the origin position to all the effector positions
            effector_pos = [jp + self.effector_positions[0] for jp in effector_pos]

            return effector_pos
        else:
            # Check that the list of joint_angles has the right length
            assert len(joint_angles) == len(self.effector_positions) - 1
            arm_lengths = self.arm_lengths

            effector_pos = [np.array([0, 0])]
            cumulative_angle = 0
            for i in range(len(arm_lengths)):
                # Working out the forward kinematics, placing the first link at the origin
                angle = joint_angles[i] + cumulative_angle
                cumulative_angle += joint_angles[i]
                calc_pos = effector_pos[-1] + arm_lengths[i] * np.array([np.cos(angle), np.sin(angle)])
                effector_pos.append(calc_pos)

            # Adding the origin position to all the effector positions
            effector_pos = [jp + self.effector_positions[0] for jp in effector_pos]

            return effector_pos

    def apply_angle_changes(self,angle_changes, initial_angles, dynamics, batch = False):
        '''
        Apply angle changes accroding to specified dynamcis.
        :param batch: Do it batch mode, where dim = 0 is the batch dimention
        :return: new_angles = initial_angles + angle_changes*initial_angles
        '''
        if(batch):
            new_angles = [(initial_angles[:,i]+angle_changes[:,i]*dynamics[:,i]).reshape(-1,1) for i in range(angle_changes.shape[1])]
            new_effector_pos = self.effector_pos_from_angles(new_angles, batch)

            return new_effector_pos, new_angles
        else:
            new_angles = [initial_angles[i] + angle_changes[i] * dynamics[i] for i in range(len(angle_changes))]
            new_effector_pos = self.effector_pos_from_angles(new_angles)

            return new_effector_pos, new_angles


    def move_joints(self, angle_changes, external_dynamics_params=None):
        '''
        Move the joints of the arm by some given angle change, subject to the effective dynamics of the arm.
        :param angle_changes: List or array with the  amount in radiants by which to move each joint.
        :param external_dynamics_params: If present, use these dynamics parameters to move the joint.
        '''
        # Verify that the  angle array given is of the correct size
        assert len(angle_changes) == len(self.effector_positions) - 1

        if (external_dynamics_params is not None):
            use_dyns = self.get_dyns_from_params(external_dynamics_params)
        else:
            use_dyns = self.__effective_dynamics

        # Get previous joint angles, add the the changes, run forward kinematics and apply the changes
        j_a = self.joint_angles()

        #j_a = [j_a[i] + angle_changes[i] * use_dyns[i] for i in range(len(angle_changes))]

        #new_join_pos = self.effector_pos_from_angles(j_a)
        new_join_pos, _ = self.apply_angle_changes(angle_changes,j_a,use_dyns)

        assert (new_join_pos[0] == self.effector_positions[0]).all()

        self.effector_positions = new_join_pos

    def modulate_angle(self, angle):
        '''Map the given angle betwwen pi and -pi'''
        twopi = 2 * np.pi
        angle = angle % twopi

        angle = (angle + twopi) % twopi

        if (angle > np.pi):
            angle -= twopi

        return angle

    def reach_check(self, goal_pos):
        '''Verify that the goal is within the reaching disk of the arm'''
        if (np.linalg.norm(goal_pos - self.origin_position) > np.sum(self.arm_lengths)):
            return False

        return True

    def full_strech_positions(self, goal_pos):
        '''Return the effector positions of a fully stretched arm in the direction of the specified goal'''
        arm_lengths = self.arm_lengths

        vec = goal_pos - self.origin_position
        dir = vec / np.linalg.norm(vec)

        positions = [self.origin_position]

        cumulative_length = 0
        for l in arm_lengths:
            cumulative_length += l
            positions.append(dir * cumulative_length + self.origin_position)

        return positions

    def fabrik(self, goal_pos, threshold=0.0000001, external_positions=None):
        '''
        Use the FABRIK algorithm solve the inverse kinematics and get the effector positions where the goal is reached.
        :param goal_pos: The goal position to reach.
        :param threshold: Threshold in terms of euclidean distance for considering the goal reached.
        :param external_positions: If external positions are supplied, use these as the arm's initial effector positions instead.
        :return: The list of effector positions such that, starting at the current (or external) positions, the arm now touches the goal.
        '''

        if (not self.reach_check(goal_pos)):
            return self.full_strech_positions(goal_pos)

        if (external_positions is None):
            positions = self.effector_positions
            arm_lengths = self.arm_lengths
        else:
            positions = external_positions
            arm_lengths = self.get_arm_lengths(external_positions)

        max_idx = len(positions) - 1
        while (np.linalg.norm(goal_pos - positions[-1]) > threshold):

            # Starting from the goal position, work backwards and calculate intermediate effector positions.
            tmp_effector_pos = collections.deque()
            tmp_effector_pos.append(goal_pos)

            for m in range(max_idx - 1, -1, -1):
                vec = positions[m] - tmp_effector_pos[0]
                tmp_pos = vec / np.linalg.norm(vec) * arm_lengths[m] + tmp_effector_pos[0]
                tmp_effector_pos.appendleft(tmp_pos)

            # Starting from the origin, adjust the effector positions.
            for m in range(1, max_idx + 1):
                vec = tmp_effector_pos[m] - positions[m - 1]
                positions[m] = vec / np.linalg.norm(vec) * arm_lengths[m - 1] + positions[m - 1]

        return positions

    def get_angle_actions(self, new_positions, initial_positions=None, iterations=1, correct_for_dynamics=False,
                          external_dynamics=None):
        '''
        Given a set of new effector positions, calculate a joint angles differential  making the arm move towards this
        new configuration.
        :param new_positions: New effector positions - List of numpy arrays of length 2
        :param initial_positions: If given, use these as the initial effector positions to calculate the angle differences.
        :param iterations: Number of times the angle differencial should be applied to reach the end position from the
        start position.
        :param correct_for_dynamics: Whether to calculate the angle differences to nullify the effect of the dynamics of the arm.
        :param external_dynamics: If supplied, treat the external dynamics as the dynamics parameters of the arm.
        :return:
        '''
        if (correct_for_dynamics):
            dyns = self.effective_dynamics
        else:
            dyns = np.array([1] * (len(self.effector_positions) - 1))

        if (external_dynamics is not None):
            dyns = self.get_dyns_from_params(external_dynamics)

        initial_j_a = self.joint_angles(initial_positions)
        final_j_a = self.joint_angles(new_positions)
        return np.array([self.modulate_angle(final_j_a[i] - initial_j_a[i]) / (iterations * dyns[i]) for i in
                         range(len(initial_j_a))])




