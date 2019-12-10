import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .model_classes import Localization, Classifier

afn = nn.LeakyReLU(1/3, inplace=True)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


### CLASSIFICATION ###

class plankton_CNN(Classifier):
    default_parameters = [32,32,64,64,128,128,128,256,256,256,512,512]

    def out(self, n):
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n, 121)
        )

    def get_layers(self, in_shape):
        mp = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)
        # should exactly halve the side of even-sided inputs
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_shape[0], self.param[0], kernel_size=(3,3), padding=1),
                afn,
            ),
            nn.Sequential(
                nn.Conv2d(self.param[0], self.param[1], kernel_size=(3,3), padding=1),
                afn, mp,
            ), # cyclic roll
            nn.Sequential(
                nn.Conv2d(self.param[1], self.param[2], kernel_size=(3,3), padding=1),
                afn,
            ),
            nn.Sequential(
                nn.Conv2d(self.param[2], self.param[3], kernel_size=(3,3), padding=1),
                afn, mp,
            ), # cyclic roll
            nn.Sequential(
                nn.Conv2d(self.param[3], self.param[4], kernel_size=(3,3), padding=1),
                afn,
            ),
            nn.Sequential(
                nn.Conv2d(self.param[4], self.param[5], kernel_size=(3,3), padding=1),
                afn,
            ),
            nn.Sequential(
                nn.Conv2d(self.param[5], self.param[6], kernel_size=(3,3), padding=1),
                afn, mp,
            ), # cyclic roll
            nn.Sequential(
                nn.Conv2d(self.param[6], self.param[7], kernel_size=(3,3), padding=1),
                afn,
            ),
            nn.Sequential(
                nn.Conv2d(self.param[7], self.param[8], kernel_size=(3,3), padding=1),
                afn,
            ),
            nn.Sequential(
                nn.Conv2d(self.param[8], self.param[9], kernel_size=(3,3), padding=1),
                afn, mp,
            ), # cyclic roll
            nn.Sequential(
                Flatten(),
                nn.Dropout(0.5),
                nn.Linear((in_shape[-1]//(2**4))**2*self.param[9], self.param[10]),
                afn,
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.param[10], self.param[11]),
                afn,
            ),
        ])