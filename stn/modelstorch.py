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
        self.parameters = [32]
        if not parameters is None:
            assert len(parameters) == len(self.parameters)
            self.parameters = parameters

    def get_layers(self, in_shape):
        return t.nn.ModuleList([
            Flatten(),
            t.nn.Linear(np.prod(in_shape), self.parameters[0]),
            afn()
        ])

class FCN_localization:
    def __init__(self, parameters = None, dropout = None):
        assert dropout is None
        self.parameters = [32,32,32]
        if not parameters is None:
            assert len(parameters) == len(self.parameters)
            self.parameters = parameters

    def get_layers(self, in_shape):
        return t.nn.ModuleList([
            Flatten(),
            t.nn.Linear(np.prod(in_shape), self.parameters[0]),
            afn(),
            t.nn.Linear(self.parameters[0], self.parameters[1]),
            afn(),
            t.nn.Linear(self.parameters[1], self.parameters[2]),
            afn()
        ])

class CNN_localization:
    def __init__(self, parameters = None, dropout = None):
        assert dropout is None
        self.parameters = [20,20,20]
        if not parameters is None:
            assert len(parameters) == len(self.parameters)
            self.parameters = parameters

    def get_layers(self, in_shape):
        final_in = self.parameters[1] * ((in_shape[1]/2-4)/2 - 4)**2
        assert final_in == int(final_in), 'Input shape not compatible with localization CNN'
        return t.nn.ModuleList([
            Downsample(), # DOWNSAMPLING
            t.nn.Conv2d(in_shape[0], self.parameters[0], kernel_size=(5,5)),
            t.nn.MaxPool2d(kernel_size=2, stride=2),
            afn(),
            t.nn.Conv2d(self.parameters[0], self.parameters[1], kernel_size=(5,5)),
            afn(),
            Flatten(),
            t.nn.Linear(int(final_in), self.parameters[2]),
            afn()
        ])

# class CNN2_localization:
#     def __init__(self, parameters = None, dropout = None):
#         self.parameters = [20,20,20]
#         self.dropout = dropout or 0
#         if not parameters is None:
#             assert len(parameters) == len(self.parameters)
#             self.parameters = parameters

#     def get_layers(self):
#         return [
#             # k.layers.Dropout(self.dropout),
#             tf.layers.Conv2D(filters=self.parameters[0], kernel_size=(5,5), activation=activation_fn),
#             tf.layers.MaxPooling2D(pool_size=2, strides=2),
#             # k.layers.Dropout(self.dropout),
#             tf.layers.Conv2D(filters=self.parameters[1], kernel_size=(5,5), activation=activation_fn),
#             tf.layers.MaxPooling2D(pool_size=2, strides=2),
#             tf.layers.Flatten(),
#             # k.layers.Dropout(self.dropout),
#             tf.layers.Dense(units=self.parameters[2], activation=activation_fn),
#         ]

def no_stn(parameters = None, dropout = None):
    assert parameters is None
    return False

# classification architecture
class FCN:
    def __init__(self, parameters = None, dropout = None):
        assert dropout is None
        self.parameters = [256,200]
        if not parameters is None:
            assert len(parameters) == len(self.parameters)
            self.parameters = parameters

    def get_layers(self, in_shape):
        return t.nn.ModuleList([
            Flatten(),
            t.nn.Linear(np.prod(in_shape), self.parameters[0]),
            afn(),
            t.nn.Linear(self.parameters[0], self.parameters[1]),
            afn(),
            t.nn.Linear(self.parameters[1], 10)
        ])

class CNN: # for mnist
    def __init__(self, parameters = None, dropout = None):
        assert dropout is None
        self.parameters = [64,64]
        if not parameters is None:
            assert len(parameters) == len(self.parameters)
            self.parameters = parameters

    def get_layers(self, in_shape):
        final_in = self.parameters[1] * (((in_shape[1]-8)/2 - 6)/2)**2
        assert final_in == int(final_in), 'Input shape not compatible with CNN'
        return t.nn.ModuleList([
            t.nn.Conv2d(in_shape[0], self.parameters[0], kernel_size = (9,9)),
            t.nn.MaxPool2d(kernel_size = 2, stride = 2),
            afn(),
            t.nn.Conv2d(self.parameters[0], self.parameters[1], kernel_size = (7,7)),
            t.nn.MaxPool2d(kernel_size = 2, stride = 2),
            afn(),
            Flatten(),
            t.nn.Linear(int(final_in), 10)
        ])

# class CNN2:
#     def __init__(self, parameters = None, dropout = None):
#         self.parameters = [64,64,64]
#         self.dropout = dropout or 0
#         if not parameters is None:
#             assert len(parameters) == len(self.parameters)
#             self.parameters = parameters

#     def get_layers(self):
#         return [
#             tf.layers.Conv2D(filters = self.parameters[0], kernel_size = (5,5), activation = activation_fn),
#             tf.layers.MaxPooling2D(pool_size = 2, strides = 2),
#             k.layers.Dropout(self.dropout),
#             tf.layers.Conv2D(filters = self.parameters[1], kernel_size = (5,5), activation = activation_fn),
#             tf.layers.MaxPooling2D(pool_size = 2, strides = 2),
#             k.layers.Dropout(self.dropout),
#             tf.layers.Conv2D(filters = self.parameters[2], kernel_size = (3,3), activation = activation_fn),
#             tf.layers.Flatten(),
#             k.layers.Dropout(self.dropout),
#             tf.layers.Dense(units = 10, activation = activation_fn),
#             tf.layers.Dense(units = 10)
#             # tf.layers.Dense(units = 10, activation = 'softmax')
#         ]

# read arguments
model_dic = {
    'FCN': FCN,
    'CNN': CNN,
    # 'CNN2': CNN2,
}

localization_dic = {
    'CNN':   CNN_localization,
    # 'CNN2':  CNN2_localization,
    'FCN':   FCN_localization,
    'small': Small_localization,
    'false': no_stn
}


# import matplotlib.pyplot as plt

class Net(t.nn.Module):
    def __init__(self, layers_obj, localization_obj, stn_placement, loop, input_shape):
        super().__init__()
        # self.layers = layers_obj.get_layers(input_shape)
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
    
    def stn(self, x, y = None):
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)
        to_transform = x if y is None else y
        # plt.imshow(to_transform.detach()[0,0,:,:])
        grid = t.nn.functional.affine_grid(theta, to_transform.size())
        transformed = t.nn.functional.grid_sample(to_transform, grid)
        # plt.figure()
        # plt.imshow(transformed.detach()[0,0,:,:])
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

        # for i,layer in enumerate(self.layers):
        #     print('Layer', i, ': ', layer)
        #     x = layer(x)
        x = self.post_stn(x)
        return x