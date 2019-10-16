#%%
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
from model_classes import Localization, Classifier, Downsample

afn = nn.ReLU

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# localization architectures
class Small_localization(Localization):
    default_parameters = [32]

    def init_model(self, in_shape):
        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(np.prod(in_shape), self.param[0]),
            afn()
        )

class ylva_localization(Localization):
    default_parameters = [16, 16, 16]

    def init_model(self, in_shape):
        self.model = nn.Sequential(
            nn.Conv2d(in_shape[0], self.param[0], kernel_size = (3,3),
                        stride= 1 if in_shape[1] <= 14 else 2),
            afn(),
            nn.Conv2d(self.param[0], self.param[1], kernel_size = (3,3), stride=2),
            afn(),
            nn.Conv2d(self.param[1], self.param[2], kernel_size = (3,3)),
            afn(),
            Flatten(),
        )

class FCN_localization(Localization):
    default_parameters = [32,32,32]

    def init_model(self, in_shape):
        self.l1 = nn.Linear(np.prod(in_shape), self.param[0])
        self.l2 = nn.Linear(self.param[0], self.param[1])
        self.l3 = nn.Linear(self.param[1], self.param[2])

    def model(self, x):
        x = F.relu(self.l1(x.view(x.size(0), -1)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x


class FCN_localization_batchnorm(Localization):
    default_parameters = [32,32,32]

    def init_model(self, in_shape):
        # Removed one batchnorm from the beginning
        self.l1 = nn.Linear(np.prod(in_shape), self.param[0])
        self.b1 = nn.BatchNorm1d(self.param[0])
        self.l2 = nn.Linear(self.param[0], self.param[1])
        self.b2 = nn.BatchNorm1d(self.param[1])
        self.l3 = nn.Linear(self.param[1], self.param[2])
        self.b3 = nn.BatchNorm1d(self.param[2])

    def model(self, x):
        x = self.b1(F.relu(self.l1(x.view(x.size(0), -1))))
        x = self.b2(F.relu(self.l2(x)))
        x = self.b3(F.relu(self.l3(x)))
        return x

class FCN_localization_maxpool(Localization):
    default_parameters = [32,32,32]

    def init_model(self, in_shape):
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b1 = nn.BatchNorm1d(np.prod(in_shape)//4)
        self.l1 = nn.Linear(np.prod(in_shape)//4, self.param[0])
        self.b2 = nn.BatchNorm1d(self.param[0])
        self.l2 = nn.Linear(self.param[0], self.param[1])
        self.b3 = nn.BatchNorm1d(self.param[1])
        self.l3 = nn.Linear(self.param[1], self.param[2])
        self.b4 = nn.BatchNorm1d(self.param[2])

    def model(self, x):
        x = self.mp(x)
        x = F.relu(self.l1(self.b1(x.view(x.size(0), -1))))
        x = F.relu(self.l2(self.b2(x)))
        x = F.relu(self.l3(self.b3(x)))
        return self.b4(x)


class CNNFCN_localization(Localization):
    '''MNIST localization consisting of jaderberg's initial convolution
    followed by the three layers from the FCN localization
    '''

    default_parameters = [64, 32, 32, 32]

    def init_model(self, in_shape):
        self.c = nn.Conv2d(in_shape[0], self.param[0], kernel_size = (9,9))
        self.mp = nn.MaxPool2d(kernel_size = 2, stride = 2)
        flattened = self.param[0] * ((in_shape[1] - 8)//2)**2
        self.l1 = nn.Linear(flattened, self.param[1])
        self.l2 = nn.Linear(self.param[1], self.param[2])
        self.l3 = nn.Linear(self.param[2], self.param[3])

    def model(self, x):
        x = F.relu(self.mp(self.c(x)))
        x = F.relu(self.l1(x.view(x.size(0), -1)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x


class CNNFCN_batchnorm(Localization):
    '''MNIST localization consisting of jaderberg's initial convolution
    followed by the three layers from the FCN localization, including the
    extra maxpool used in FCNmp.
    '''

    default_parameters = [64, 32, 32, 32]

    def init_model(self, in_shape):
        self.c = nn.Conv2d(in_shape[0], self.param[0], kernel_size = (9,9))
        self.mp = nn.MaxPool2d(kernel_size = 4, stride = 4)
        flattened = self.param[0] * ((in_shape[1] - 8)//4)**2
        self.b1 = nn.BatchNorm1d(flattened)
        self.l1 = nn.Linear(flattened, self.param[1])
        self.b2 = nn.BatchNorm1d(self.param[1])
        self.l2 = nn.Linear(self.param[1], self.param[2])
        self.b3 = nn.BatchNorm1d(self.param[2])
        self.l3 = nn.Linear(self.param[2], self.param[3])
        self.b4 = nn.BatchNorm1d(self.param[3])

    def model(self, x):
        x = F.relu(self.mp(self.c(x)))
        x = F.relu(self.l1(self.b1(x.view(x.size(0), -1))))
        x = F.relu(self.l2(self.b2(x)))
        x = F.relu(self.l3(self.b3(x)))
        return self.b4(x)


class CNN_localization(Localization):
    default_parameters = [20,20,20]

    def init_model(self, in_shape):
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5))
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5))
        side = int((in_shape[-1]/2 - 4)/2 - 4)
        self.l = nn.Linear(self.param[1] * side**2, self.param[2])

    def model(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x = self.mp(F.relu(self.c1(x)))
        x = F.relu(self.c2(x))
        x = F.relu(self.l(x.view(x.size(0), -1)))
        return x

class CNN_localization_batchnorm(Localization):
    default_parameters = [20,20,20]

    def init_model(self, in_shape):
        # Removed one batchnorm from the beginning
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5))
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b1 = nn.BatchNorm2d(self.param[0])
        self.c2 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5))
        flattened = self.param[1] * int((in_shape[-1]/2 - 4)/2 - 4)**2
        self.b2 = nn.BatchNorm1d(flattened)
        self.l = nn.Linear(flattened, self.param[2])
        self.b3 = nn.BatchNorm1d(self.param[2])

    def model(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x = self.b1(self.mp(F.relu(self.c1(x))))
        x = self.b2(F.relu(self.c2(x)))
        x = self.b3(F.relu(self.l(x.view(x.size(0), -1))))
        return x


class CNN_middleloc_batchnorm(Localization):
    default_parameters = [24,48,48]

    def init_model(self, in_shape):
        # Removed one batchnorm from the beginning
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5))
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        flattened = self.param[0] * (3 if in_shape[-1] == 10 else 5)**2
        self.b1 = nn.BatchNorm1d(flattened)
        self.l1 = nn.Linear(flattened, self.param[1])
        self.b2 = nn.BatchNorm1d(self.param[1])
        self.l2 = nn.Linear(self.param[1], self.param[2])
        self.b3 = nn.BatchNorm1d(self.param[2])

    def model(self, x):
        if x.size(-1) == 28:
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x = self.b1(self.mp(F.relu(self.c1(x))))
        x = self.b2(F.relu(self.l1(x.view(x.size(0), -1))))
        x = self.b3(F.relu(self.l2(x)))
        return x


class CNN_middleloc(Localization):
    default_parameters = [20,20,20]

    def init_model(self, in_shape):
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5))
        if in_shape[-1] > 24:
            self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5))
        side = 8 if in_shape[1] == 28 else 2
        self.l = nn.Linear(self.param[1] * side**2, self.param[2])

    def model(self, x):
        x = self.c1(x)
        if x.size(-1) > 20:
            x = self.mp(x)
        x = F.relu(x)
        x = F.relu(self.c2(x))
        x = F.relu(self.l(x.view(x.size(0), -1)))
        return x


class CNN_translate(Localization):
    default_parameters = [20,20,20]

    def init_model(self, in_shape):
        self.c1 = nn.Conv2d(in_shape[0], self.param[0], kernel_size=(5,5))
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = nn.Conv2d(self.param[0], self.param[1], kernel_size=(5,5))
        if in_shape[-1] > 30:
            self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        side = 12 if in_shape[-1] == 60 else 7
        self.l = nn.Linear(self.param[1] * side**2, self.param[2])

    def model(self, x):
        x = self.mp(F.relu(self.c1(x)))
        x = F.relu(self.c2(x))
        if x.size(-1) > 20: # the size should be 7 or 24
            x = self.mp(x)
        x = F.relu(self.l(x.view(x.size(0), -1)))
        return x


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


# classification architectures
class FCN(Classifier):
    default_parameters = [256,200]

    def out(self,n):
        return nn.Linear(n, 10)

    def get_layers(self, in_shape):
        return nn.ModuleList([
            Flatten(),
            nn.Linear(np.prod(in_shape), self.param[0]),
            afn(),
            nn.Linear(self.param[0], self.param[1]),
            afn(),
        ])

class CNN(Classifier): # original for mnist, works for cifar
    default_parameters = [64,64]

    def out(self,n):
        return nn.Linear(n, 10)

    def get_layers(self, in_shape, downsample=None):
        return nn.ModuleList([
            nn.Conv2d(in_shape[0], self.param[0], kernel_size = (9,9)),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            afn(),
            nn.Conv2d(self.param[0], self.param[1], kernel_size = (7,7)),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            afn(),
        ])

class CNN_batchnorm(Classifier): # original for mnist, works for cifar
    default_parameters = [64,64]

    def out(self,n):
        return nn.Linear(n, 10)

    def get_layers(self, in_shape, downsample=None):
        # Changed to put batchnorm after each layer instead of before
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_shape[0], self.param[0], kernel_size = (9,9)),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                afn(),
                nn.BatchNorm2d(self.param[0]),
            ),
            nn.Sequential(
                nn.Conv2d(self.param[0], self.param[1], kernel_size = (7,7)),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                afn(),
                nn.BatchNorm2d(self.param[1]),
            ),
        ])

class ylva_mnist(Classifier): # ylva uses adam, lr 0.003
    default_parameters = [16, 16, 32, 32, 100]

    def out(self,n):
        return nn.Sequential(
            nn.Linear(n, self.param[4]),
            afn(),
            nn.Dropout(0.15),
            nn.Linear(self.param[4], 10)
        )

    def get_layers(self, in_shape, downsample=None):
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_shape[0], self.param[0], kernel_size = (3,3)),
                afn(),
                nn.BatchNorm2d(self.param[0]),
            ),
            nn.Sequential(
                nn.Conv2d(self.param[0], self.param[1], kernel_size = (3,3), stride=2),
                afn(),
                nn.BatchNorm2d(self.param[1]),
            ),
            nn.Sequential(
                nn.Conv2d(self.param[1], self.param[2], kernel_size = (3,3)),
                afn(),
                nn.BatchNorm2d(self.param[2]),
            ),
            nn.Sequential(
                nn.Conv2d(self.param[2], self.param[3], kernel_size = (3,3), stride=2),
                afn(),
                nn.BatchNorm2d(self.param[3]),
            ),
            # nn.Conv2d(self.param[3], self.param[4], 4, stride=1),
            # afn(),
            # nn.Dropout(0.15),
            # nn.Conv2d(self.param[4], 10, 1, stride=1),
        ])

class CNN2(Classifier): # for cifar
    default_parameters = [32,32,64,64,128,128]

    def out(self,n):
        return nn.Linear(n, 10)

    def get_layers(self, in_shape, downsample=None):
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

    def get_layers(self, in_shape, downsample=None):
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


# dictionaries for mapping arguments to classes
localization_dict = {
    'CNN':   CNN_localization,
    'CNNm':  CNN_middleloc,
    'CNNmb': CNN_middleloc_batchnorm,
    'CNNt':  CNN_translate,
    'CNNb':  CNN_localization_batchnorm,
    'CNN2':  CNN_localization2,
    'FCN':   FCN_localization,
    'FCNb':  FCN_localization_batchnorm,
    'FCNmp': FCN_localization_maxpool,
    'CNNFCN':CNNFCN_localization,
    'CNNFCNb':CNNFCN_batchnorm,
    'ylva':  ylva_localization,
    'small': Small_localization,
    'SVHN-l':SVHN_large,
    'SVHN-d':SVHN_dropout,
    'SVHN-s':SVHN_small,
    'false': False,
}

model_dict = {
    'FCN': FCN,
    'CNN': CNN,
    'CNNb': CNN_batchnorm,
    'ylva': ylva_mnist,
    'CNN2': CNN2,
    'SVHN-CNN': SVHN_CNN,
}