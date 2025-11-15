import torch
import torch.nn as nn

class malwareDetector(nn.Module):
    def __init__(self, inputDim=2381, hiddenDims=[512, 256], dropout=0.2):
        super(malwareDetector, self).__init__()
        layers = []

        layers.append(nn.Linear(inputDim, hiddenDims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for i in range(len(hiddenDims)-1):
            layers.append(nn.Linear(hiddenDims[i],hiddenDims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hiddenDims[-1],1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self,x):
        return self.network(x)