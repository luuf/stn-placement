import tensorflow as tf
import numpy as np
from transformer import spatial_transformer_network as transformer
from functools import reduce

activation_fn = tf.nn.relu

# localization architecture
class Small_localization:
    def __init__(self, parameters = None):
        self.parameters = [32]
        if not parameters is None:
            assert len(parameters) == len(self.parameters)
            self.parameters = parameters

    def get_layers(self):
        return [
            tf.layers.Dense(units = self.parameters[0], activation = activation_fn),
        ]

class FCN_localization:
    def __init__(self, parameters = None):
        self.parameters = [32,32,32]
        if not parameters is None:
            assert len(parameters) == len(self.parameters)
            self.parameters = parameters

    def get_layers(self):
        return [
            tf.layers.Flatten(),
            tf.layers.Dense(units = self.parameters[0], activation = activation_fn),
            tf.layers.Dense(units = self.parameters[1], activation = activation_fn),
            tf.layers.Dense(units = self.parameters[2], activation = activation_fn),
        ]

class CNN_localization:
    def __init__(self, parameters = None):
        self.parameters = [20,20,20]
        if not parameters is None:
            assert len(parameters) == len(self.parameters)
            self.parameters = parameters

    def get_layers(self):
        return [
            tf.layers.Conv2D(filters=self.parameters[0], kernel_size=(5,5), activation=activation_fn),
            tf.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.layers.Conv2D(filters=self.parameters[1], kernel_size=(5,5), activation=activation_fn),
            tf.layers.Flatten(),
            tf.layers.Dense(units=self.parameters[2], activation=activation_fn),
        ]

def no_stn(parameters = None):
    assert parameters is None
    return False

# classification architecture
class FCN:
    def __init__(self, parameters = None):
        self.parameters = [256,200]
        if not parameters is None:
            assert len(parameters) == len(self.parameters)
            self.parameters = parameters

    def get_layers(self, parameters = None):
        return [
            tf.layers.Flatten(),
            tf.layers.Dense(units = self.parameters[0], activation = activation_fn),
            tf.layers.Dense(units = self.parameters[1], activation = activation_fn),
            tf.layers.Dense(units = 10)
            # tf.layers.Dense(units = 10, activation = 'softmax')
        ]

class CNN:
    def __init__(self, parameters = None):
        self.parameters = [64,64]
        if not parameters is None:
            assert len(parameters) == len(self.parameters)
            self.parameters = parameters

    def get_layers(self):
        return [
            tf.layers.Conv2D(filters = self.parameters[0], kernel_size = (9,9), activation = activation_fn),
            tf.layers.MaxPooling2D(pool_size = 2, strides = 2),
            tf.layers.Conv2D(filters = self.parameters[1], kernel_size = (7,7), activation = activation_fn),
            tf.layers.MaxPooling2D(pool_size = 2, strides = 2),
            tf.layers.Flatten(),
            tf.layers.Dense(units = 10)
            # tf.layers.Dense(units = 10, activation = 'softmax')
        ]


# read arguments
model_dic = {
    'CNN': CNN,
    'FCN': FCN
}

localization_dic = {
    'CNN':   CNN_localization,
    'FCN':   FCN_localization,
    'small': Small_localization,
    'false': no_stn
}


def sequential(layers, initial):
    return reduce(lambda l0,l1: l1(l0), layers, initial)

class STN(tf.keras.layers.Layer):
    def call(self, inputs):
        return transformer(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def build(self, input_shape):
        pass
    
    def get_config(self):
        return None

    @classmethod
    def from_config(cls, config):
        return STN()

def compose_model(layers_obj, localization_obj, stn_placement, loop, shape):
    layers = layers_obj.get_layers()
    inp = tf.keras.layers.Input(shape=shape)

    if localization_obj:
        # stn = tf.keras.layers.Lambda(lambda inputs: transformer(inputs[0],inputs[1]))
        stn = STN()

        first_layers = layers[:stn_placement]
        localization_in = sequential(first_layers, inp)
        # first_layers = tf.keras.layers.Lambda(lambda i: sequential(layers[:stn_placement], i))
        # localization_in = first_layers(inp)

        parameter_in = sequential(localization_obj.get_layers(), localization_in)
        parameters = tf.layers.Dense(
            units = 6,
            kernel_initializer = tf.keras.initializers.Zeros(),
            bias_initializer = tf.keras.initializers.Constant([1,0,0,1,0,0])
        )(parameter_in)

        if loop:
            first_out = sequential(first_layers, stn([inp, parameters]))
            # first_out = first_layers(stn([inp, parameters]))
        else:
            first_out = stn([localization_in, parameters])

        pred = sequential(layers[stn_placement:], first_out)
    else:
        pred = sequential(layers, inp)

    return tf.keras.models.Model(inputs=inp, outputs=pred)
        
