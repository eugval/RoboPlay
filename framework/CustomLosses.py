import torch.nn as nn
import torch

class CombinedWeightedLosses(nn.Module):
    def __init__(self,losses, weights):
        super(CombinedWeightedLosses,self).__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, outputs, labels):
        ls = []

        for i in range(len(labels)):
            ls.append(self.losses[i](outputs[i],labels[i]))

        initial = torch.mul(ls[0],self.weights[0])
        total = torch.add(initial,self.weights[1],ls[1])
        for i in range(2,len(ls)):
            total =torch.add(total,self.weights[i])

        return total


    def update_weights(self,weights):
        assert len(weights) == len(self.weights)

        self.weights = weights