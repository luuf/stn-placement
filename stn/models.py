#%%
import torch as t
import torch.nn.functional as F
import numpy as np
from functools import reduce
# import utils


afn = t.nn.ReLU


def get_output_shape(input_shape, module):
    dummy = t.tensor( # pylint: disable=not-callable
        np.zeros([1]+list(input_shape),
        dtype='float32')
    )
    out = module(dummy)
    return out.shape[1:]


class Flatten(t.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Downsample(t.nn.Module):
    def forward(self, input):
        return F.interpolate(
            input,
            scale_factor=0.5,
            mode='bilinear'
        )


class Modular_Model(t.nn.Module):
    def __init__(self, parameters):
        super().__init__()

        if not (parameters is None or parameters == []):
            assert len(parameters) == len(self.default_parameters)
            self.param = parameters 
        else:
            self.param = self.default_parameters


class Localization(Modular_Model):
    def __init__(self, parameters, input_shape):
        super().__init__(parameters)

        self.model = t.nn.Sequential(*self.get_layers(input_shape))

        out_shape = get_output_shape(input_shape, self.model)
        assert len(out_shape) == 1, "Localization output must be flat"
        self.affine_param = t.nn.Linear(out_shape[0], 6)
        self.affine_param.weight.data.zero_()
        self.affine_param.bias.data.copy_(t.tensor([1,0,0,0,1,0],dtype=t.float)) # pylint: disable=no-member,not-callable

    def forward(self, x):
        return self.affine_param(self.model(x))


class Classifier(Modular_Model):
    def __init__(self, parameters, input_shape, localization_class,
                 localization_parameters, stn_placement, loop, data_tag):
        super().__init__(parameters)

        layers = self.get_layers(input_shape)
        if len(stn_placement) > 0:
            split_at = zip([0]+stn_placement[:-1],stn_placement)
            self.pre_stn = t.nn.ModuleList([t.nn.Sequential(*layers[s:e]) for s,e in split_at])
            self.final_layers = t.nn.Sequential(*layers[stn_placement[-1]:])
        else:
            self.pre_stn = []
            self.final_layers = layers

        if loop:
            self.loop_models = t.nn.ModuleList(
                    [t.nn.Sequential(*self.pre_stn[:i]) for i in range(1,len(self.pre_stn)+1)])
            self.register_buffer(
                'base_theta',
                t.tensor(np.identity(3, dtype=np.float32)) # pylint: disable=not-callable
            )
            # I need to define theta as a tensor before forward,
            # so that it's automatically ported it to device with model
        else:
            self.loop_models = None

        if localization_class:
            shape = input_shape
            self.localization = t.nn.ModuleList()
            for model in self.pre_stn:
                shape = get_output_shape(shape, model)
                self.localization.append(localization_class(localization_parameters, shape))
        else:
            self.localization = None

        if data_tag in ['translate','clutter']:
            print('STN will downsample, since the data is', data_tag)

            assert not stn_placement or len(stn_placement) == 1, \
                'Have not implemented downsample for multiple stns'

            self.size_transform = np.array([1,1,2,2])
            if loop:
                final_shape = get_output_shape(input_shape, t.nn.Sequential(
                    Downsample(), *self.pre_stn, self.post_stn
                ))
            else:
                final_shape = get_output_shape(input_shape, t.nn.Sequential(
                    *self.pre_stn, Downsample(), self.final_layers
                ))
        else:
            self.size_transform = np.array([1,1,1,1])
            final_shape = get_output_shape(input_shape, t.nn.Sequential(
                *self.pre_stn, self.final_layers
            ))

        self.output = self.out(np.prod(final_shape))

        # self.layers = layers_obj.get_layers(input_shape) # FOR DEBUGGING
    
    def stn(self, theta, y):
        theta = theta.view(-1, 2, 3)
        size = np.array(y.shape) // self.size_transform
        grid = F.affine_grid(theta, t.Size(size))
        transformed = F.grid_sample(y, grid)
        # plt.imshow(transformed.detach()[0,0,:,:])
        # plt.figure()
        # plt.imshow(to_transform.detach()[0,0,:,:])
        # plt.show()
        return transformed
    
    def forward(self, x):
        if self.localization:
            if self.loop_models:
                input_image = x
                theta = self.base_theta
                for l,m in zip(self.localization,self.loop_models):
                    theta = F.pad(l(m(x)), (0,3)).view((-1,3,3)) * theta
                    x = self.stn(theta[:,0:2], input_image)
                x = m(x)
            else:
                for l,m in zip(self.localization,self.pre_stn):
                    localization_input = m(x)
                    x = self.stn(l(localization_input), localization_input)

        # for i,layer in enumerate(self.layers):  # FOR DEBUGGING
        #     print('Layer', i, ': ', layer)
        #     print('Shape', x.shape)
        #     x = layer(x)
        x = self.final_layers(x)

        return self.output(x.view(x.size(0),-1))


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
                t.nn.Dropout2d(0.5),
                # no dropout in first layer
            ),
            t.nn.Sequential(
                t.nn.Conv2d(self.param[0], self.param[1], kernel_size = (5,5), padding=2),
                afn(),
                t.nn.Dropout2d(0.5),
            ),
            t.nn.Sequential(
                t.nn.Conv2d(self.param[1], self.param[2], kernel_size = (5,5), padding=2),
                afn(),
                t.nn.MaxPool2d(kernel_size = 2, stride = 2),
                t.nn.Dropout2d(0.5),
            ),
            t.nn.Sequential(
                t.nn.Conv2d(self.param[2], self.param[3], kernel_size = (5,5), padding=2),
                afn(),
                t.nn.Dropout2d(0.5),
            ),
            t.nn.Sequential(
                t.nn.Conv2d(self.param[3], self.param[4], kernel_size = (5,5), padding=2),
                afn(),
                t.nn.MaxPool2d(kernel_size = 2, stride = 2),
                t.nn.Dropout2d(0.5),
            ),
            t.nn.Sequential(
                t.nn.Conv2d(self.param[4], self.param[5], kernel_size = (5,5), padding=2),
                afn(),
                t.nn.Dropout2d(0.5),
            ),
            t.nn.Sequential(
                t.nn.Conv2d(self.param[5], self.param[6], kernel_size = (5,5), padding=2),
                afn(),
                t.nn.MaxPool2d(kernel_size = 2, stride = 2),
                t.nn.Dropout2d(0.5),
            ),
            t.nn.Sequential(
                t.nn.Conv2d(self.param[6], self.param[7], kernel_size = (5,5), padding=2),
                afn(),
                t.nn.Dropout2d(0.5),
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
