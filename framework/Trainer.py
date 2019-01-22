from abc import ABC, abstractmethod
import numpy as np
import torch.nn as nn
import torch.optim as optim


class Trainer(ABC):

    @abstractmethod
    def train(self):
        '''Main training method'''
        pass

    def dagger_function(self,rollout_function, utils, data_buffer, world, train_data,
                        epoch,randomising_dynamics, batch_size,
                         dagger_start, dagger_discard, dagger_add_interval, dagger_rollouts
                        ):
        if (epoch >= dagger_start):
            if (epoch == dagger_start):
                print('Starting dagger...')
                if (dagger_discard):
                    print('emptying dataset...')
                    data_buffer.reset()

            if ((epoch - dagger_start) % dagger_add_interval == 0):
                print('Adding dagger data...')
                dagger_acc, dagger_soft_acc, states, actions, dynamics, trajectory_indices = utils.do_rollouts(
                    rollout_function,
                    dagger_rollouts, report_trajectories=True, randomise_dynamics=randomising_dynamics
                    )

                print('Dagger Acc : {}, Dagger min dist: {}'.format(dagger_acc, dagger_soft_acc))
                data_buffer.add_policy_generated_data(world, states, actions, dynamics, trajectory_indices)

        report_interval = np.ceil(len(train_data) / (batch_size * 10))
        return report_interval

    def get_criterion(self, criterion_str,criterion_params):
        if(criterion_str == 'MSE'):
            criterion = nn.MSELoss(**criterion_params)
        else:
            raise ValueError('Loss ill defined')
        return criterion

    def get_optimiser(self,optimiser_str,optimiser_params, params_to_optimise):
        if (optimiser_str == 'Adam'):
            optimiser = optim.Adam(params_to_optimise, **optimiser_params)
        elif(optimiser_str == 'Adadelta'):
            optimiser = optim.Adadelta(params_to_optimise, **optimiser_params)
        elif (optimiser_str == 'SGD'):
            optimiser = optim.SGD(params_to_optimise,**optimiser_params)
        else:
            raise ValueError('Optimiser ill defined')

        return optimiser


