import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
from copy import deepcopy


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
    dummy = torch.tensor(
        np.zeros([2]+list(input_shape),  # batchnorm requires batchsize >1
        dtype='float32')
    )
    out = module(dummy)
    return out.shape[1:]


class Downsample(nn.Module):
    """Halves each side of the input by downsampling bilinearly"""
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')


class Modular_Model(nn.Module):
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
    def __init__(self, parameters, input_shape, llr=1):
        super().__init__(parameters)

        self.init_model(input_shape)

        out_shape = get_output_shape(input_shape, self.model)
        assert len(out_shape) == 1, "Localization output must be flat"
        self.affine_param = nn.Linear(out_shape[0], 6)
        self.affine_param.weight.data.zero_()
        self.affine_param.bias.data.zero_()
        self.hook = None if llr == 1 else lambda x: x*llr
        self.register_buffer(
            'identity', torch.tensor([1,0,0,0,1,0],dtype=torch.float))

    def forward(self, x):
        x = self.affine_param(self.model(x))
        if self.hook and x.requires_grad: # needed to avoid problems during testing
            x.register_hook(self.hook)
        return x + self.identity


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
                 localization_parameters, stn_placement, loop, data_tag,
                 batchnorm=False, deep=False, iterative=True, downsample=False):
        super().__init__(parameters)

        if data_tag in ['translate','clutter'] and localization_class:
            scale_down_by = 2
        elif data_tag.split('/')[-1] == 'unprocessed_' and localization_class:
            scale_down_by = 1
        else:
            scale_down_by = 1
        assert scale_down_by == 1 or loop or len(stn_placement) == 1
        self.size_transform = np.array([1,1,scale_down_by,scale_down_by])
        downsampler = Downsample(1/scale_down_by)

        downsampled_shape = (input_shape[0], input_shape[1]//scale_down_by, input_shape[2]//scale_down_by)
        layers = self.get_layers(downsampled_shape)
            
        if len(stn_placement) > 0:
            split_at = zip([0]+stn_placement[:-1],stn_placement)
            self.pre_stn = nn.ModuleList([nn.Sequential(*layers[s:e]) for s,e in split_at])
            self.final_layers = nn.Sequential(*layers[stn_placement[-1]:])
        else:
            self.pre_stn = nn.ModuleList([])
            self.final_layers = nn.Sequential(*layers)

        if loop:
            self.loop_models = nn.ModuleList(
                    [nn.Sequential(*self.pre_stn[:i]) for i in range(1,len(self.pre_stn)+1)])
        else:
            self.loop_models = None

        if localization_class:
            if not batchnorm:
                self.batchnorm = False
            elif loop:
                self.batchnorm = nn.ModuleList(
                    [nn.BatchNorm2d(input_shape[0], affine=False) for _ in self.pre_stn])
            else:
                self.batchnorm = nn.ModuleList()
                # batchnorms with appropriate shapes are added in next loop
            shape = input_shape
            self.localization = nn.ModuleList()
            for i,model in enumerate(self.pre_stn):
                if iterative and len(model) == 0 and i > 0:
                    self.localization.append(self.localization[-1])
                else:
                    shape = get_output_shape(shape, model)
                    if deep:
                        self.localization.append(nn.Sequential(
                            *deepcopy(self.pre_stn[:i+1]),
                            localization_class(localization_parameters, shape)
                        ))
                    else:
                        self.localization.append(
                            localization_class(localization_parameters, shape))
                    if batchnorm and not loop:
                        self.batchnorm.append(nn.BatchNorm2d(shape[0], affine=False))
                if i == 0 and loop and scale_down_by != 1:
                    shape = get_output_shape(downsampled_shape, model)
        else:
            self.localization = None

        if loop:
            final_shape = get_output_shape(input_shape, nn.Sequential(
                downsampler, *self.pre_stn, self.final_layers
            ))
        else:
            final_shape = get_output_shape(input_shape, nn.Sequential(
                *self.pre_stn, downsampler, self.final_layers
            ))

        if data_tag in ['translate', 'clutter', 'mnist', 'scale']: # and (stn_placement == [0] or loop):
            self.padding_mode = 'border'
        elif 'plankton' in data_tag.split('/'):
            self.padding_mode = 'border'
        else:
            self.padding_mode = 'zeros'
        print('padding mode', self.padding_mode)

        if deep:
            self.final_layers = nn.Sequential(
                *self.pre_stn, self.final_layers)
            self.pre_stn = nn.ModuleList(nn.Sequential() for _ in self.pre_stn)

        self.output = self.out(np.prod(final_shape))

        self.register_buffer(
            'base_theta',
            torch.tensor(np.identity(3, dtype=np.float32))
        )
        # I need to define theta as a tensor before forward, so that
        # it's automatically ported to device together with model

        #self.downsampler = downsampler

        # self.layers = layers_obj.get_layers(input_shape) # FOR DEBUGGING
    
    def stn(self, theta, y):
        theta = theta.view(-1, 2, 3)
        size = np.array(y.shape) // self.size_transform
        grid = F.affine_grid(theta, torch.Size(size))
        transformed = F.grid_sample(y, grid, padding_mode=self.padding_mode)
        # plt.imshow(transformed.detach()[0,0,:,:])
        # plt.figure()
        # plt.imshow(to_transform.detach()[0,0,:,:])
        # plt.show()
        return transformed
    
    def forward(self, x):
        if self.localization:
            theta = self.base_theta
            if self.loop_models:
                input_image = x
                #x = self.downsampler(x)
                #x = self.stn(theta.repeat(x.size(0),1,1)[:,0:2,:], x)
                for i,m in enumerate(self.loop_models):
                    localization_output = self.localization[i](m(x))
                    mat = F.pad(localization_output, (0,3)).view((-1,3,3))
                    mat[:,2,2] = 1
                    theta = torch.matmul(theta,mat)
                    # note that the new transformation is multiplied
                    # from the right. Since the parameters are the
                    # inverse of the parameters that would be applied
                    # to the numbers, this yields the same parameters
                    # that would result from each transformation being
                    # applied after the previous, with the stn.
                    # Empirically, there's no noticeably difference
                    # between multiplying from the right and left.
                    x = self.stn(theta[:,0:2,:], input_image)
                    if self.batchnorm:
                        x = self.batchnorm[i](x)
                if hasattr(self, 'pretrain') and self.pretrain:
                    return theta[:,0:2,:]
                x = m(x)
            else:
                for i,m in enumerate(self.pre_stn):
                    if i == 0 or len(m) > 0:
                        loc_input = m(x)
                        loc_output = self.localization[i](loc_input)
                        theta = self.base_theta
                    else:
                        loc_output = self.localization[i](x)
                    mat = F.pad(loc_output, (0,3)).view((-1,3,3))
                    mat[:,2,2] = 1
                    theta = torch.matmul(theta,mat)
                    x = self.stn(theta[:,0:2,:], loc_input)
                    if self.batchnorm:
                        x = self.batchnorm[i](x)

        # for i,layer in enumerate(self.layers):  # FOR DEBUGGING
        #     print('Layer', i, ': ', layer)
        #     print('Shape', x.shape)
        #     x = layer(x)
        x = self.final_layers(x)

        return self.output(x.view(x.size(0),-1))

    def add_iteration(self):
        assert not self.batchnorm
        if self.loop_models:
            self.loop_models.append(self.loop_models[-1])
        else:
            self.pre_stn.append(nn.Sequential())
        self.localization.append(self.localization[-1])
