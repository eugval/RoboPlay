import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP,self).__init__()

        self.input_size = layers[0]
        self.output_size = layers[-1]

        mlp = []


        for i in range(len(layers)-2):
            mlp.append(nn.Linear(layers[i],layers[i+1]))
            mlp.append(nn.BatchNorm1d(layers[i+1]))
            mlp.append(nn.ReLU(True))


        mlp.append(nn.Linear(layers[-2],layers[-1]))

        self.net = nn.Sequential(*mlp)


    def forward(self,x):
        return self.net(x)
