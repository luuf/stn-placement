import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .model_classes import Localization, Classifier

afn = nn.ReLU

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


### LOCALIZATION ###

class SVHN_small(Localization):
    default_parameters = [32,32]

    def init_model(self, in_shape):
        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(np.prod(in_shape), self.param[0]),
            afn(),
            nn.Linear(self.param[0], self.param[1]),
            afn(),
        )

class SVHN_large(Localization):
    default_parameters = [32,32,32,32]

    def init_model(self, in_shape):
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5), padding=2)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5), padding=2)
        side = in_shape[1]/2
        assert side == int(side)
        self.l1 = nn.Linear(self.param[1] * int(side)**2, self.param[2])
        self.l2 = nn.Linear(self.param[2], self.param[3])

    def model(self, x):
        x = self.mp(F.relu(self.c1(x)))
        x = F.relu(self.c2(x))
        x = F.relu(self.l1(x.view(x.size(0), -1)))
        x = F.relu(self.l2(x))
        return x

class SVHN_dropout(Localization):
    default_parameters = [32,32,32,32]

    def init_model(self, in_shape):
        self.droupout = nn.Dropout(0.5)
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5), padding=2)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5), padding=2)
        side = in_shape[1]/2
        assert side == int(side)
        self.l1 = nn.Linear(self.param[1] * int(side)**2, self.param[2])
        self.l2 = nn.Linear(self.param[2], self.param[3])

    def model(self, x):
        x = self.dropout(self.mp(F.relu(self.c1(x))))
        x = self.dropout(F.relu(self.c2(x)))
        x = self.dropout(F.relu(self.l1(x.view(x.size(0), -1))))
        x = self.dropout(F.relu(self.l2(x)))
        return x


### CLASSIFICATION ###

class SVHN_CNN(Classifier):
    default_parameters = [48,64,128,160,192,192,192,192,3072,3072,3072]

    class out(nn.Module):
        def __init__(self, neurons):
            super().__init__()
            self.out_layers = nn.ModuleList([
                nn.Linear(neurons,11) for i in range(5)
            ])
        def forward(self, x):
            return [layer(x) for layer in self.out_layers]

    def get_layers(self, in_shape):
        final_side = in_shape[1]/2/2/2/2
        assert final_side == int(final_side), 'Input shape not compatible with localization CNN'
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_shape[0], self.param[0], kernel_size = (5,5), padding=2),
                afn(),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                # no dropout in first layer
            ),
            nn.Sequential(
                nn.Conv2d(self.param[0], self.param[1], kernel_size = (5,5), padding=2),
                afn(),
                nn.Dropout(0.5),
            ),
            nn.Sequential(
                nn.Conv2d(self.param[1], self.param[2], kernel_size = (5,5), padding=2),
                afn(),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                nn.Dropout(0.5),
            ),
            nn.Sequential(
                nn.Conv2d(self.param[2], self.param[3], kernel_size = (5,5), padding=2),
                afn(),
                nn.Dropout(0.5),
            ),
            nn.Sequential(
                nn.Conv2d(self.param[3], self.param[4], kernel_size = (5,5), padding=2),
                afn(),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                nn.Dropout(0.5),
            ),
            nn.Sequential(
                nn.Conv2d(self.param[4], self.param[5], kernel_size = (5,5), padding=2),
                afn(),
                nn.Dropout(0.5),
            ),
            nn.Sequential(
                nn.Conv2d(self.param[5], self.param[6], kernel_size = (5,5), padding=2),
                afn(),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                nn.Dropout(0.5),
            ),
            nn.Sequential(
                nn.Conv2d(self.param[6], self.param[7], kernel_size = (5,5), padding=2),
                afn(),
                nn.Dropout(0.5),
            ),
            Flatten(),
            nn.Linear(self.param[7] * int(final_side)**2, self.param[8]),
            afn(),
            nn.Dropout(0.5),
            nn.Linear(self.param[8], self.param[9]),
            afn(),
            nn.Dropout(0.5),
            nn.Linear(self.param[9], self.param[10]),
            afn(),
            nn.Dropout(0.5),
        ])
