import torch as t
import torch.nn.functional as F
import numpy as np
from functools import reduce


def get_output_shape(input_shape, module):
    """Takes an input_shape and a module, and returns the shape that
    the module would return given a tensor of shape input_shape.
    Assumes that the module's output shape only depends on the shape of
    the input.

    Args:
        input_shape: any iterable that describes the shape. Shouldn't
            include any batch_size.
        module: anything that inherits from torch.nn.module
    """
    dummy = t.tensor(
        np.zeros([1]+list(input_shape),
        dtype='float32')
    )
    out = module(dummy)
    return out.shape[1:]


class Downsample(t.nn.Module):
    """Halves each side of the input by downsampling bilinearly"""
    def forward(self, x):
        return F.interpolate(x, scale_factor=0.5, mode='bilinear')


class Modular_Model(t.nn.Module):
    """Superclass that handles some optional parameters of modules.
    All subclasses are required to define default_parameters, which are
    used if no parameters are passed.

    Args:
        parameters: a list of the number of filters or neurons for each
            layer, or None.
    """
    def __init__(self, parameters):
        super().__init__()

        if not (parameters is None or parameters == []):
            assert len(parameters) == len(self.default_parameters)
            self.param = parameters 
        else:
            self.param = self.default_parameters


class Localization(Modular_Model):
    """Superclass for affine localization networks. Subclasses should
    implement a function init_model and a model self.model that
    does most of the computations. The final layer, that gets
    the 6 affine parameters, are defined here, and shouldn't be
    included in subclasses.

    Args:
        parameters (list or None): are passed to the Modular_Model
            superclass.
        input_shape: any iterable that describes the shape of the input
            to the network. Shouldn't include any batch size.
    """
    def __init__(self, parameters, input_shape):
        super().__init__(parameters)

        self.init_model(input_shape)

        out_shape = get_output_shape(input_shape, self.model)
        assert len(out_shape) == 1, "Localization output must be flat"
        self.affine_param = t.nn.Linear(out_shape[0], 6)
        self.affine_param.weight.data.zero_()
        self.affine_param.bias.data.copy_(t.tensor([1,0,0,0,1,0],dtype=t.float))

    def forward(self, x):
        return self.affine_param(self.model(x))


class Classifier(Modular_Model):
    """Superclass for classification networks. Subclasses should
    implement a function get_layers that returns a list of all layers
    that the classification network contains, and a function out that
    takes the output from the final layer of get_layers and returns the
    final classification vector.

    Args:
        parameters (list or None): are passed to the Modular_Model
            superclass.
        input_shape: any iterable that describes the shape of the input
            to the network. Shouldn't include any batch size.
        localization_class: A subclass of Localization if the network
            uses an STN. Otherwise None or False.
        localization_parameters (list or None): are passed to the
            Modular_Model superclass of the localization network.
        stn_placement (list): contains the indices of each layer from
            get_layers that an STN should be placed before. Should be
            empty if no STN is used.
        loop (bool): True if an STN with looping is used, otherwise
            False.
        data_tag (string): Name of the dataset, used for some
            transforms that should only happen for a few datasets.
    """

    def __init__(self, parameters, input_shape, localization_class,
                 localization_parameters, stn_placement, loop, data_tag):
        super().__init__(parameters)

        layers = self.get_layers(input_shape)
        if len(stn_placement) > 0:
            split_at = zip([0]+stn_placement[:-1],stn_placement)
            self.pre_stn = t.nn.ModuleList([t.nn.Sequential(*layers[s:e]) for s,e in split_at])
            self.final_layers = t.nn.Sequential(*layers[stn_placement[-1]:])
        else:
            self.pre_stn = t.nn.ModuleList([])
            self.final_layers = t.nn.Sequential(*layers)

        if loop:
            self.loop_models = t.nn.ModuleList(
                    [t.nn.Sequential(*self.pre_stn[:i]) for i in range(1,len(self.pre_stn)+1)])
            self.register_buffer(
                'base_theta',
                t.tensor(np.identity(3, dtype=np.float32))
            )
            # I need to define theta as a tensor before forward, so that
            # it's automatically ported to device together with model
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
                    Downsample(), *self.pre_stn, self.final_layers
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
        transformed = F.grid_sample(y, grid, padding_mode='border') # CHANGE THIS
        print('STN using border padding')
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
                    localization_output = l(m(x))
                    mat = F.pad(localization_output, (0,3)).view((-1,3,3))
                    mat[:,2,2] = 1
                    theta = t.matmul(theta,mat)
                    # note that the new transformation is multiplied
                    # from the right. Since the parameters are the
                    # inverse of the parameters that would be applied
                    # to the numbers, this yields the same parameters
                    # that would result from each transformation being
                    # applied after the previous, with the stn.
                    # Empirically, there's no noticeably difference
                    # between multiplying from the right and left.
                    x = self.stn(theta[:,0:2,:], input_image)
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
