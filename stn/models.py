#%%
import torch as t
import numpy as np
from functools import reduce
# import utils

afn = t.nn.ReLU

#%%
def get_output_shape(input_shape, module):
    dummy = t.tensor(np.zeros([1]+list(input_shape), dtype='float32')) # pylint: disable=not-callable
    out = module(dummy)
    return out.shape[1:]
#%%
class Flatten(t.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Downsample(t.nn.Module):
    def forward(self, input):
        return t.nn.functional.interpolate(
            input,
            scale_factor=0.5,
            mode='bilinear'
        )

# localization architecture
class Small_localization:
    def __init__(self, parameters = None, dropout = None):
        assert dropout is None
        self.param = [32]
        if not parameters is None:
            assert len(parameters) == len(self.param)
            self.param = parameters

    def get_layers(self, in_shape):
        return t.nn.ModuleList([
            Flatten(),
            t.nn.Linear(np.prod(in_shape), self.param[0]),
            afn()
        ])

class FCN_localization:
    def __init__(self, parameters = None, dropout = None):
        assert dropout is None
        self.param = [32,32,32]
        if not parameters is None:
            assert len(parameters) == len(self.param)
            self.param = parameters

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

class CNN_localization:
    def __init__(self, parameters = None, dropout = None):
        assert dropout is None
        self.param = [20,20,20]
        if not parameters is None:
            assert len(parameters) == len(self.param)
            self.param = parameters

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

class CNN_localization2: # for cifar
    def __init__(self, parameters = None, dropout = None):
        assert dropout is None
        self.param = [20,40,80]
        if not parameters is None:
            assert len(parameters) == len(self.param)
            self.param = parameters

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

# class CNN2_localization:
#     def __init__(self, parameters = None, dropout = None):
#         self.param = [20,20,20]
#         self.dropout = dropout or 0
#         if not parameters is None:
#             assert len(parameters) == len(self.param)
#             self.param = parameters

#     def get_layers(self):
#         return [
#             # k.layers.Dropout(self.dropout),
#             tf.layers.Conv2D(filters=self.param[0], kernel_size=(5,5), activation=activation_fn),
#             tf.layers.MaxPooling2D(pool_size=2, strides=2),
#             # k.layers.Dropout(self.dropout),
#             tf.layers.Conv2D(filters=self.param[1], kernel_size=(5,5), activation=activation_fn),
#             tf.layers.MaxPooling2D(pool_size=2, strides=2),
#             tf.layers.Flatten(),
#             # k.layers.Dropout(self.dropout),
#             tf.layers.Dense(units=self.param[2], activation=activation_fn),
#         ]

def no_stn(parameters = None, dropout = None):
    assert parameters is None
    return False

# classification architecture
class FCN:
    def __init__(self, parameters = None, dropout = None):
        assert dropout is None
        self.param = [256,200]
        if not parameters is None:
            assert len(parameters) == len(self.param)
            self.param = parameters

    def get_layers(self, in_shape):
        return t.nn.ModuleList([
            Flatten(),
            t.nn.Linear(np.prod(in_shape), self.param[0]),
            afn(),
            t.nn.Linear(self.param[0], self.param[1]),
            afn(),
        ])

class CNN: # original for mnist, works for cifar
    def __init__(self, parameters = None, dropout = None):
        assert dropout is None
        self.param = [64,64]
        if not parameters is None:
            assert len(parameters) == len(self.param)
            self.param = parameters

    def get_layers(self, in_shape, downsample=None):
        return t.nn.ModuleList([
            t.nn.Conv2d(in_shape[0], self.param[0], kernel_size = (9,9)),
            t.nn.MaxPool2d(kernel_size = 2, stride = 2),
            afn(),
            t.nn.Conv2d(self.param[0], self.param[1], kernel_size = (7,7)),
            t.nn.MaxPool2d(kernel_size = 2, stride = 2),
            afn(),
        ])

class CNN2: # for cifar
    def __init__(self, parameters = None, dropout = None):
        assert dropout is None
        self.param = [32,32,64,64,128,128]
        if not parameters is None:
            assert len(parameters) == len(self.param)
            self.param = parameters

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

# read arguments
model_dict = {
    'FCN': FCN,
    'CNN': CNN,
    'CNN2': CNN2,
}

localization_dict = {
    'CNN':   CNN_localization,
    'CNN2':  CNN_localization2,
    # 'CNN2':  CNN2_localization,
    'FCN':   FCN_localization,
    'small': Small_localization,
    'false': no_stn
}


# import matplotlib.pyplot as plt

class Net(t.nn.Module):
    def __init__(self, layers_obj, localization_obj, stn_placement, loop, input_shape):
        super().__init__()

        layers = layers_obj.get_layers(input_shape)
        self.pre_stn = t.nn.Sequential(*layers[:stn_placement])
        self.post_stn = t.nn.Sequential(*layers[stn_placement:])
        self.loop = loop

        if localization_obj:
            in_shape = get_output_shape(input_shape, self.pre_stn)
            localization = t.nn.Sequential(*localization_obj.get_layers(in_shape))
            out_shape = get_output_shape(in_shape, localization)
            assert len(out_shape) == 1, "Localization output must be flat"
            parameters = t.nn.Linear(out_shape[0], 6)
            parameters.weight.data.zero_()
            parameters.bias.data.copy_(t.tensor([1,0,0,0,1,0],dtype=t.float)) # pylint: disable=not-callable,no-member
            self.localization = t.nn.Sequential(localization, parameters)
        else:
            self.localization = None

        if input_shape[-1] > 40 and localization_obj:
            print("STN will downsample, since the width is",input_shape[-1])
            self.downsample = Downsample()
            if loop:
                final_shape = get_output_shape(input_shape, t.nn.Sequential(
                    self.downsample, self.pre_stn, self.post_stn
                ))
            else:
                final_shape = get_output_shape(input_shape, t.nn.Sequential(
                    self.pre_stn, self.downsample, self.post_stn
                ))
        else:
            self.downsample = None
            final_shape = get_output_shape(input_shape, t.nn.Sequential(
                self.pre_stn, self.post_stn
            ))

        self.out = t.nn.Linear(np.prod(final_shape), 10)
        
        # self.layers = layers_obj.get_layers(input_shape) # FOR DEBUGGING
    
    def stn(self, x, y = None):
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)
        to_transform = x if y is None else y
        grid = t.nn.functional.affine_grid(theta, to_transform.size())
        transformed = t.nn.functional.grid_sample(to_transform, grid)
        if self.downsample:
            transformed = self.downsample(transformed)
        # plt.imshow(transformed.detach()[0,0,:,:])
        # plt.figure()
        # plt.imshow(to_transform.detach()[0,0,:,:])
        # plt.show()
        return transformed
    
    def forward(self, x):
        if self.localization:
            if self.loop:
                x = self.stn(self.pre_stn(x), x)
                x = self.pre_stn(x)
            else:
                x = self.pre_stn(x)
                x = self.stn(x)

        # for i,layer in enumerate(self.layers):  # FOR DEBUGGING
        #     print('Layer', i, ': ', layer)
        #     print('Shape', x.shape)
        #     x = layer(x)
        x = self.post_stn(x)

        return self.out(x.view(x.size(0),-1))