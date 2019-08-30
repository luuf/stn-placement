#%%
import torch as t
import torch.nn.functional as F
import numpy as np
from functools import reduce
from model_classes import Localization, Classifier, Downsample

afn = t.nn.ReLU

class Flatten(t.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# localization architectures
class Small_localization(Localization):
    default_parameters = [32]

    def get_layers(self, in_shape):
        return t.nn.ModuleList([
            Flatten(),
            t.nn.Linear(np.prod(in_shape), self.param[0]),
            afn()
        ])

class FCN_localization(Localization):
    default_parameters = [32,32,32]

    def get_layers(self, in_shape):
        return t.nn.ModuleList([
            Flatten(),
            t.nn.Linear(np.prod(in_shape), self.param[0]),
            afn(),
            t.nn.Linear(self.param[0], self.param[1]),
            afn(),
            t.nn.Linear(self.param[1], self.param[2]),
            afn()
        ])

class CNNFCN_localization(Localization): # mnist loc that emulates a looped layer without sharing parameters
    default_parameters = [64, 32, 32, 32]

    def get_layers(self, in_shape):
        return t.nn.ModuleList([
        t.nn.Conv2d(in_shape[0], self.param[0], kernel_size = (9,9)),
        t.nn.MaxPool2d(kernel_size = 2, stride = 2),
        afn(),
        Flatten(),
        t.nn.Linear(np.prod(in_shape), self.param[1]),
        afn(),
        t.nn.Linear(self.param[1], self.param[2]),
        afn(),
        t.nn.Linear(self.param[2], self.param[3]),
        afn()
    ])

class CNN_localization(Localization):
    default_parameters = [20,20,20]

    def get_layers(self, in_shape):
        final_in = self.param[1] * ((in_shape[1]/2-4)/2 - 4)**2
        assert final_in == int(final_in), 'Input shape not compatible with localization CNN'
        return t.nn.ModuleList([
            Downsample(), # DOWNSAMPLING
            t.nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5)),
            t.nn.MaxPool2d(kernel_size=2, stride=2),
            afn(),
            t.nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5)),
            afn(),
            Flatten(),
            t.nn.Linear(int(final_in), self.param[2]),
            afn()
        ])

class CNN_localization2(Localization): # for cifar
    default_parameters = [20,40,80]

    def get_layers(self, in_shape):
        final_in = self.param[1] * (((in_shape[1])/2)/2)**2
        assert final_in == int(final_in), 'Input shape not compatible with localization CNN'
        return t.nn.ModuleList([
            t.nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5), padding=2),
            afn(),
            t.nn.BatchNorm2d(self.param[0]),
            t.nn.MaxPool2d(kernel_size=2, stride=2),
            t.nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5), padding=2),
            afn(),
            t.nn.BatchNorm2d(self.param[1]),
            t.nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            t.nn.Linear(int(final_in), self.param[2]),
            afn()
        ])

class SVHN_large(Localization):
    default_parameters = [32,32,32,32]

    def get_layers(self, in_shape):
        final_side = in_shape[1]/2
        assert final_side == int(final_side)
        return t.nn.ModuleList([
            t.nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5), padding=2),
            afn(),
            t.nn.MaxPool2d(kernel_size=2, stride=2),
            t.nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5), padding=2),
            afn(),
            Flatten(),
            t.nn.Linear(self.param[1] * int(final_side)**2, self.param[2]),
            afn(),
            t.nn.Linear(self.param[2], self.param[3]),
            afn(),
        ])

class SVHN_small(Localization):
    default_parameters = [32,32]

    def get_layers(self, in_shape):
        return t.nn.ModuleList([
            Flatten(),
            t.nn.Linear(np.prod(in_shape), self.param[0]),
            afn(),
            t.nn.Linear(self.param[0], self.param[1]),
            afn(),
        ])


# classification architectures
class FCN(Classifier):
    default_parameters = [256,200]

    def out(self,n):
        return t.nn.Linear(n, 10)

    def get_layers(self, in_shape):
        return t.nn.ModuleList([
            Flatten(),
            t.nn.Linear(np.prod(in_shape), self.param[0]),
            afn(),
            t.nn.Linear(self.param[0], self.param[1]),
            afn(),
        ])

class CNN(Classifier): # original for mnist, works for cifar
    default_parameters = [64,64]

    def out(self,n):
        return t.nn.Linear(n, 10)

    def get_layers(self, in_shape, downsample=None):
        return t.nn.ModuleList([
            t.nn.Conv2d(in_shape[0], self.param[0], kernel_size = (9,9)),
            t.nn.MaxPool2d(kernel_size = 2, stride = 2),
            afn(),
            t.nn.Conv2d(self.param[0], self.param[1], kernel_size = (7,7)),
            t.nn.MaxPool2d(kernel_size = 2, stride = 2),
            afn(),
        ])

class CNN2(Classifier): # for cifar
    default_parameters = [32,32,64,64,128,128]

    def out(self,n):
        return t.nn.Linear(n, 10)

    def get_layers(self, in_shape, downsample=None):
        return t.nn.ModuleList([
            t.nn.Conv2d(in_shape[0], self.param[0], kernel_size = (3,3), padding=1),
            afn(),
            t.nn.BatchNorm2d(self.param[0]),
            t.nn.Conv2d(self.param[0], self.param[1], kernel_size = (3,3), padding=1),
            afn(),
            t.nn.BatchNorm2d(self.param[1]),
            t.nn.MaxPool2d(kernel_size = 2, stride = 2),
            t.nn.Dropout2d(0.2),

            t.nn.Conv2d(self.param[1], self.param[2], kernel_size = (3,3), padding=1),
            afn(),
            t.nn.BatchNorm2d(self.param[2]),
            t.nn.Conv2d(self.param[2], self.param[3], kernel_size = (3,3), padding=1),
            afn(),
            t.nn.BatchNorm2d(self.param[3]),
            t.nn.MaxPool2d(kernel_size = 2, stride = 2),
            t.nn.Dropout2d(0.3),

            t.nn.Conv2d(self.param[3], self.param[4], kernel_size = (3,3), padding=1),
            afn(),
            t.nn.BatchNorm2d(self.param[4]),
            t.nn.Conv2d(self.param[4], self.param[5], kernel_size = (3,3), padding=1),
            afn(),
            t.nn.BatchNorm2d(self.param[5]),
            t.nn.MaxPool2d(kernel_size = 2, stride = 2),
            t.nn.Dropout2d(0.4),
        ])


class SVHN_CNN(Classifier):
    default_parameters = [48,64,128,160,192,192,192,192,3072,3072,3072]

    class out(t.nn.Module):
        def __init__(self, neurons):
            super().__init__()
            self.out_layers = t.nn.ModuleList([
                t.nn.Linear(neurons,11) for i in range(5)
            ])
        def forward(self, x):
            return [layer(x) for layer in self.out_layers]

    def get_layers(self, in_shape, downsample=None):
        final_side = in_shape[1]/2/2/2/2
        assert final_side == int(final_side), 'Input shape not compatible with localization CNN'
        return t.nn.ModuleList([
            t.nn.Sequential(
                t.nn.Conv2d(in_shape[0], self.param[0], kernel_size = (5,5), padding=2),
                afn(),
                t.nn.MaxPool2d(kernel_size = 2, stride = 2),
                # no dropout in first layer
            ),
            t.nn.Sequential(
                t.nn.Conv2d(self.param[0], self.param[1], kernel_size = (5,5), padding=2),
                afn(),
                t.nn.Dropout(0.5),
            ),
            t.nn.Sequential(
                t.nn.Conv2d(self.param[1], self.param[2], kernel_size = (5,5), padding=2),
                afn(),
                t.nn.MaxPool2d(kernel_size = 2, stride = 2),
                t.nn.Dropout(0.5),
            ),
            t.nn.Sequential(
                t.nn.Conv2d(self.param[2], self.param[3], kernel_size = (5,5), padding=2),
                afn(),
                t.nn.Dropout(0.5),
            ),
            t.nn.Sequential(
                t.nn.Conv2d(self.param[3], self.param[4], kernel_size = (5,5), padding=2),
                afn(),
                t.nn.MaxPool2d(kernel_size = 2, stride = 2),
                t.nn.Dropout(0.5),
            ),
            t.nn.Sequential(
                t.nn.Conv2d(self.param[4], self.param[5], kernel_size = (5,5), padding=2),
                afn(),
                t.nn.Dropout(0.5),
            ),
            t.nn.Sequential(
                t.nn.Conv2d(self.param[5], self.param[6], kernel_size = (5,5), padding=2),
                afn(),
                t.nn.MaxPool2d(kernel_size = 2, stride = 2),
                t.nn.Dropout(0.5),
            ),
            t.nn.Sequential(
                t.nn.Conv2d(self.param[6], self.param[7], kernel_size = (5,5), padding=2),
                afn(),
                t.nn.Dropout(0.5),
            ),
            Flatten(),
            t.nn.Linear(self.param[7] * int(final_side)**2, self.param[8]),
            afn(),
            t.nn.Dropout(0.5),
            t.nn.Linear(self.param[8], self.param[9]),
            afn(),
            t.nn.Dropout(0.5),
            t.nn.Linear(self.param[9], self.param[10]),
            afn(),
            t.nn.Dropout(0.5),
        ])


# dictionaries for mapping arguments to classes
localization_dict = {
    'CNN':   CNN_localization,
    'CNN2':  CNN_localization2,
    'FCN':   FCN_localization,
    'CNNFCN':CNNFCN_localization,
    'small': Small_localization,
    'SVHN-l':SVHN_large,
    'SVHN-s':SVHN_small,
    'false': False,
}

model_dict = {
    'FCN': FCN,
    'CNN': CNN,
    'CNN2': CNN2,
    'SVHN-CNN': SVHN_CNN,
}
