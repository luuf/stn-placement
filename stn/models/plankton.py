import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .model_classes import Localization, Classifier

afn = nn.LeakyReLU(1/3, inplace=True)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


### LOCALIZATION ###

class plankton_large(Localization):
    default_parameters = [32,32,32,32]

    def init_model(self, in_shape):
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(3,3), padding=1)
        self.mp = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0) #changed
        self.c2 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(3,3), padding=1)

        side = in_shape[1]//2//2
        self.l1 = nn.Linear(self.param[1] * side**2, self.param[2])
        self.l2 = nn.Linear(self.param[2], self.param[3])

    def model(self, x):
        x = self.mp(F.relu(self.c1(x)))
        x = self.mp(F.relu(self.c2(x)))
        x = F.relu(self.l1(x.view(x.size(0), -1)))
        x = F.relu(self.l2(x))
        return x


class plankton_depth4(Localization):
    default_parameters = [32,32,64,64, 64,64]

    def init_model(self, in_shape):
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(3,3), padding=1)
        self.c2 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(3,3), padding=1)
        self.c3 = nn.Conv2d(self.param[1], self.param[2], kernel_size=(3,3), padding=1)
        self.c4 = nn.Conv2d(self.param[2], self.param[3], kernel_size=(3,3), padding=1)
        self.mp = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0)
        side = in_shape[-1] // 2 // 2
        self.l1 = nn.Linear(self.param[3] * side**2, self.param[4])
        self.l2 = nn.Linear(self.param[4], self.param[5])

    def model(self, x):
        x = afn(self.c1(x))
        x = self.mp(afn(self.c2(x)))
        x = afn(self.c3(x))
        x = self.mp(afn(self.c4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x

class plankton_depth2(Localization):
    default_parameters = [32,32, 64,64]

    def init_model(self, in_shape):
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(3,3), padding=1)
        self.c2 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(3,3), padding=1)
        self.mp = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0)
        side = in_shape[-1] // 2
        self.l1 = nn.Linear(self.param[1] * side**2, self.param[2])
        self.l2 = nn.Linear(self.param[2], self.param[3])

    def model(self, x):
        x = afn(self.c1(x))
        x = self.mp(afn(self.c2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x


### CLASSIFICATION ###

class plankton_CNN(Classifier):
    default_parameters = [32,32,64,64,128,128,128,256,256,256,512,512]

    def out(self, n):
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n, 121)
        )

    def get_layers(self, in_shape):
        mp = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0) #changed
        #dropout2d = nn.Dropout2d(0.2, inplace=False)
        # should exactly halve the side of even-sided inputs
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_shape[0], self.param[0], kernel_size=(3,3), padding=1),
                afn,#dropout2d
            ),
            nn.Sequential(
                nn.Conv2d(self.param[0], self.param[1], kernel_size=(3,3), padding=1),
                afn, mp,#dropout2d
            ), # cyclic roll
            nn.Sequential(
                nn.Conv2d(self.param[1], self.param[2], kernel_size=(3,3), padding=1),
                afn,#dropout2d
            ),
            nn.Sequential(
                nn.Conv2d(self.param[2], self.param[3], kernel_size=(3,3), padding=1),
                afn, mp,#dropout2d
            ), # cyclic roll
            nn.Sequential(
                nn.Conv2d(self.param[3], self.param[4], kernel_size=(3,3), padding=1),
                afn,#dropout2d
            ),
            nn.Sequential(
                nn.Conv2d(self.param[4], self.param[5], kernel_size=(3,3), padding=1),
                afn,#dropout2d
            ),
            nn.Sequential(
                nn.Conv2d(self.param[5], self.param[6], kernel_size=(3,3), padding=1),
                afn, mp,#dropout2d
            ), # cyclic roll
            nn.Sequential(
                nn.Conv2d(self.param[6], self.param[7], kernel_size=(3,3), padding=1),
                afn,#dropout2d
            ),
            nn.Sequential(
                nn.Conv2d(self.param[7], self.param[8], kernel_size=(3,3), padding=1),
                afn,#dropout2d
            ),
            nn.Sequential(
                nn.Conv2d(self.param[8], self.param[9], kernel_size=(3,3), padding=1),
                afn, mp,#dropout2d
            ), # cyclic roll
            nn.Sequential(
                Flatten(),
                nn.Dropout(0.5),
                nn.Linear((in_shape[-1]//2//2//2//2)**2*self.param[9], self.param[10]),
                afn,
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.param[10], self.param[11]),
                afn,
            ),
        ])
