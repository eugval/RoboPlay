import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from framework.OptData import OptData
import numpy as np


class DaggerTrainer(object):
    def __init__(self, world, agent, data_buffer, utils, device, number_of_input_states, save_folder,
                 batch_size=32, epochs=10, dagger_start=-1, dagger_discard=False,
                 dagger_rollouts=100,dagger_add_interval=1):
        '''
        Class for training an agent to control the robot arm.
        :param world: The world.
        :param agent: The controller.
        :param data_buffer: The data buffer.
        :param utils: The utils for this system.
        :param device: [string], 'cpu' or cuda device number (c.f. pytorch documentation)
        :param number_of_input_states: [int] (minimum value:1) The number of stacked states the agent will receive
        as an input.
        :param save_folder: [path] The folder to save the model checkpoints.
        :param batch_size: [int]
        :param epochs: [int]
        :param dagger_start: [int] The epoch after which the Dagger algorithm kicks in. If negative, the training proceeds
        without using dagger.
        :param dagger_discard: [bool] Whether to discard all the previous data once Dagger kicks in.
        :param dagger_rollouts: [int] Number of trajectories to add the the dataset at each Dagger iteration.
        :param dagger_add_interval: [int] The frequency over which to add dagger data.
        '''

        # System attributes
        self.world = world
        self.agent = agent
        self.device = device
        self.utils = utils
        self.data_buffer = data_buffer


        # Training attributes
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_data = OptData(data_buffer, utils, number_of_input_states)
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size,
                                                        shuffle=True, num_workers=2)


        self.criterion = nn.MSELoss()
        self.optimiser = optim.Adam(agent.network.parameters(),lr=0.0001)
        #self.optimiser=torch.optim.SGD(agent.network.parameters(), lr=0.1, momentum=0.9)


        # Dagger attributes
        if (dagger_start < 0):
            self.dagger_start = epochs + 1
        else:
            self.dagger_start = dagger_start

        self.dagger_rollouts = dagger_rollouts

        self.dagger_discard = dagger_discard
        self.dagger_add_interval = dagger_add_interval

        #saving params
        if (not os.path.exists(save_folder)):
            raise ValueError('An issue arised with the path to the saving folder')
        else:
            self.save_folder = save_folder


    def train(self):
        print('Training the agent...')
        save_folder = self.save_folder

        cumulative_loss = 0.0
        gradient_steps = 0
        report_interval = np.ceil(len(self.train_data)/(self.batch_size*10))
        train_losses = []
        train_losses_per_epoch = []
        test_loss_per_epoch = []
        train_acc_per_epoch = []
        train_min_dist_per_epoch = []
        test_acc_per_epoch = []
        test_min_dist_per_epoch = []

        epochs = self.epochs
        dagger_start  = self.dagger_start

        for epoch in range(epochs):
            for data in self.train_loader:
                inputs = data['state_sequence'].to(self.device)
                labels = data['optimal_action_t'].to(self.device)

                if(inputs.size()[0] == 1):
                    continue

                self.optimiser.zero_grad()

                outputs = self.agent.network(inputs)
                loss = self.criterion(outputs,labels)
                loss.backward()
                self.optimiser.step()

                cumulative_loss += loss.item()
                gradient_steps += 1

                if(gradient_steps % report_interval == 0):
                    l = cumulative_loss / gradient_steps
                    print('epoch :{}, gradient step : {}, loss : {}'.format(epoch, gradient_steps, l))
                    train_losses.append((gradient_steps, l))

            l = cumulative_loss/gradient_steps
            train_losses_per_epoch.append(l)

            #force 200 training examples for train accuracy
            prop = 200 /self.train_data.data_buffer.number_of_trajectories
            if(prop>=1):
                prop = 1.0

            train_accuracy, train_soft_accuracy = self.utils.do_rollouts( self.rollout_function,
                initial_configs = self.data_buffer.initial_configs, subset_proportion= prop)

            print('Epoch {}, Train Accuracy : {}, Train Average min distance : {}'.format(epoch, train_accuracy,
                                                                                          train_soft_accuracy))
            train_acc_per_epoch.append(train_accuracy)
            train_min_dist_per_epoch.append(train_soft_accuracy)

            test_accuracy, test_soft_acc = self.utils.do_rollouts( self.rollout_function,
                                                                   initial_configs=self.data_buffer.test_initial_configs,
                                                                   subset_proportion=1.0)
            print('Epoch {}, Test Accuracy : {}, Test average min distance: {}'.format(epoch, test_accuracy,
                                                                                       test_soft_acc))
            test_acc_per_epoch.append(test_accuracy)
            test_min_dist_per_epoch.append(test_soft_acc)

            # Save training checkpoint
            checkpoint_path = os.path.join(save_folder,'agent_{}.pth'.format(epoch))
            torch.save({
                'epoch':epoch,
                'model_state_dict':self.agent.network.state_dict(),
                'optimiser_state_dict': self.optimiser.state_dict(),
                'loss': l,
                'test_accuracy':test_accuracy,
                'test_soft_accuracy':test_soft_acc,
                'train_accuracy':train_accuracy,
                'train_soft_accuracy': train_soft_accuracy
            },checkpoint_path)


            if(epoch>= dagger_start):
                if(epoch == dagger_start):
                    print('Starting dagger...')
                    if(self.dagger_discard):
                        print('emptying dataset...')
                        self.data_buffer.reset()

                if((epoch - dagger_start) % self.dagger_add_interval ==0):
                    print('Adding dagger data...')
                    dagger_acc , dagger_soft_acc, states,actions,dynamics,trajectory_indices = self.utils.do_rollouts(
                        self.rollout_function, self.dagger_rollouts, report_trajectories= True,
                    )

                    print('Dagger Acc : {}, Dagger min dist: {}'.format(dagger_acc,dagger_soft_acc))

                    self.data_buffer.add_policy_generated_data(self.world,states,actions,dynamics,trajectory_indices)

                    report_interval = np.ceil(len(self.train_data)/(self.batch_size*10))


        training_history = {
            'loss': train_losses,
            'loss_per_epoch':train_losses_per_epoch,
            'test_loss_per_epoch':test_loss_per_epoch,
             'train_accuracy':train_acc_per_epoch,
            'train_min_dist': train_min_dist_per_epoch,
            'test_accuracy':test_acc_per_epoch,
            'test_min_dist':test_min_dist_per_epoch
        }

        history_path = os.path.join(save_folder,'agent_training_history.pckl')
        pickle.dump(training_history,open(history_path,'wb'))

        data_buffer_file_path = os.path.join(save_folder,'agent_training_data_buffer.pckl')
        self.data_buffer.save(data_buffer_file_path)

        return training_history




    def rollout_function(self, **kwargs):
        state = kwargs['state']
        state = self.agent.expand_input_for_inference(state, self.world, self.utils.standardise_data)
        return self.agent.infer(state)



