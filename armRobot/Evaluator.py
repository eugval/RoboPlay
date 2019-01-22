import os
from framework.Evaluator import BaseEvaluator
import numpy as np
import matplotlib.pyplot as plt
import math

class Evaluator(BaseEvaluator):
    def __init__(self, data_buffer, model_folder):
        super(Evaluator,self).__init__(data_buffer,model_folder)

    def plot_training_history(self, training_history, stage = 'agent', show = False):
        gradient_steps, loss = zip(*training_history['loss'])
        loss_per_epoch = training_history['loss_per_epoch']
        test_loss_per_epoch = training_history['test_loss_per_epoch']
        test_accuracy = training_history['test_accuracy']
        train_accuracy = training_history['train_accuracy']
        train_soft_acc = training_history['train_min_dist']
        test_soft_acc = training_history['test_min_dist']

        save_folder = self.model_folder



        plt.figure()
        plt.plot(gradient_steps,loss)
        plt.xlabel('gradient steps')
        plt.ylabel('loss')
        plt.title('{} training loss vs gradient steps'.format(stage))
        plt.savefig(os.path.join(save_folder,'{}_loss_hist.png'.format(stage)))


        plt.figure()
        plt.plot(np.arange(len(loss_per_epoch)), loss_per_epoch, label = 'train loss', color ='b')
        plt.plot(np.arange(len(test_loss_per_epoch)), test_loss_per_epoch, label = 'test loss', color ='r')
        plt.legend(loc='upper right')
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title('losses vs epochs')
        plt.savefig(os.path.join(save_folder, '{}_loss_per_epoch.png'.format(stage)))


        plt.figure()
        plt.plot(np.arange(len(test_accuracy)),test_accuracy, label = 'test accuracy', color ='r')
        plt.plot(np.arange(len(train_accuracy)), train_accuracy,label = 'train accuracy', color ='b')
        plt.legend(loc='upper right')
        plt.ylim(0, 1)
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('{} Accuracy vs epochs'.format(stage))
        plt.savefig(os.path.join(save_folder, '{}_accuracy.png'.format(stage)))



        plt.figure()
        plt.plot(np.arange(len(train_soft_acc)), train_soft_acc,label = 'train soft accuracy', color ='b')
        plt.plot(np.arange(len(test_soft_acc)), test_soft_acc,label = 'test softaccuracy', color ='r')
        plt.yscale('log')
        plt.legend(loc='upper right')
        plt.xlabel('epochs')
        plt.ylabel('Train min Euclidean distance')
        plt.title('{} Minimum distance to target vs epochs'.format(stage))
        plt.savefig(os.path.join(save_folder, '{}_soft_accuracy_.png'.format(stage)))



        if(show):
            plt.show()
        else:
            plt.close('all')

    def plot_states(self, proportion = 0.1, title_prefix ='agent', show = False):
        '''
        Randomly choose a subset of the states in the buffer and plot the full trajectories  containing those states.
        :param proportion: [float between 0 and 1]Proportion of states to choose
        :param title_prefix: [string] Title prefix
        '''
        states_len = self.data_buffer.states_len()-1
        number_of_trials = math.ceil(proportion * states_len)
        indx_array = np.random.choice(states_len, number_of_trials)
        states_to_plot = []

        for idx in indx_array:
            states_to_plot.append(self.data_buffer.get_state(idx)[:2])
            idx_f = idx + 1
            idx_b = idx

            traj_idx = self.data_buffer.get_trajectory_index(idx_f)
            while(traj_idx !=0):
                states_to_plot.append(self.data_buffer.get_state(idx_f)[:2])
                idx_f +=1
                try:
                    traj_idx = self.data_buffer.get_trajectory_index(idx_f)
                except IndexError:
                    break

            traj_idx = self.data_buffer.get_trajectory_index(idx_b)
            while (traj_idx != 0):
                idx_b -= 1
                states_to_plot.append(self.data_buffer.get_state(idx_b)[:2])
                traj_idx = self.data_buffer.get_trajectory_index(idx_b)


        if (len(states_to_plot) == 0):
            plt.scatter([], [])
        else:
            plt.scatter(*zip(*states_to_plot), s=1)
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.xlabel('x_position')
        plt.ylabel('y_position')
        plt.title('{} end effector distribution'.format(title_prefix))
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])

        save_folder = os.path.join(self.model_folder, 'state_distributions')
        if (not os.path.exists(save_folder)):
            os.mkdir(save_folder)
        plt.savefig(os.path.join(save_folder,'{}_State_dist.png'.format(title_prefix)))
        if (show):
            plt.show()
        else:
            plt.close('all')