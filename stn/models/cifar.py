import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .model_classes import Localization, Classifier

afn = nn.ReLU

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


### LOCALIZATION ###

class CNN_localization2(Localization): # for cifar
    default_parameters = [20,40,80]

    def init_model(self, in_shape):
        final_in = self.param[1] * (((in_shape[1])/2)/2)**2
        assert final_in == int(final_in), 'Input shape not compatible with localization CNN'
        self.model = nn.Sequential(
            nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5), padding=2),
            afn(),
            nn.BatchNorm2d(self.param[0]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5), padding=2),
            afn(),
            nn.BatchNorm2d(self.param[1]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(int(final_in), self.param[2]),
            afn()
        )


### CLASSIFICATION ###

class CNN2(Classifier): # for cifar
    default_parameters = [32,32,64,64,128,128]

    def out(self,n):
        return nn.Linear(n, 10)

    def get_layers(self, in_shape):
        return nn.ModuleList([
            nn.Conv2d(in_shape[0], self.param[0], kernel_size = (3,3), padding=1),
            afn(),
            nn.BatchNorm2d(self.param[0]),
            nn.Conv2d(self.param[0], self.param[1], kernel_size = (3,3), padding=1),
            afn(),
            nn.BatchNorm2d(self.param[1]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(0.2),

            nn.Conv2d(self.param[1], self.param[2], kernel_size = (3,3), padding=1),
            afn(),
            nn.BatchNorm2d(self.param[2]),
            nn.Conv2d(self.param[2], self.param[3], kernel_size = (3,3), padding=1),
            afn(),
            nn.BatchNorm2d(self.param[3]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(0.3),

            nn.Conv2d(self.param[3], self.param[4], kernel_size = (3,3), padding=1),
            afn(),
            nn.BatchNorm2d(self.param[4]),
            nn.Conv2d(self.param[4], self.param[5], kernel_size = (3,3), padding=1),
            afn(),
            nn.BatchNorm2d(self.param[5]),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(0.4),
        ])
