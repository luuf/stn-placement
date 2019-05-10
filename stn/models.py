import tensorflow as tf
import numpy as np
from stn import spatial_transformer_network as transformer

activation_fn = tf.nn.relu

#% Architecture
def small_localization():
    return [
        tf.layers.Dense(units = 32, activation = activation_fn),
    ]

def FCN_localization():
    return [
        tf.layers.Flatten(),
        tf.layers.Dense(units = 32, activation = activation_fn),
        tf.layers.Dense(units = 32, activation = activation_fn),
        tf.layers.Dense(units = 32, activation = activation_fn),
    ]

def CNN_localization():
    return [
        tf.layers.Conv2D(filters=20, kernel_size=(5,5), activation=activation_fn),
        tf.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.layers.Conv2D(filters=20, kernel_size=(5,5), activation=activation_fn),
        tf.layers.Flatten(),
        tf.layers.Dense(units=20, activation=activation_fn),
    ]

def FCN():
    return [
        tf.layers.Flatten(),
        tf.layers.Dense(units = 256, activation = activation_fn),
        tf.layers.Dense(units = 200, activation = activation_fn),
        tf.layers.Dense(units = 10)
        # tf.layers.Dense(units = 10, activation = 'softmax')
    ]

def CNN():
    return [
        tf.layers.Conv2D(filters = 64, kernel_size = (9,9), activation = activation_fn),
        tf.layers.MaxPooling2D(pool_size = 2, strides = 2),
        tf.layers.Conv2D(filters = 64, kernel_size = (7,7), activation = activation_fn),
        tf.layers.MaxPooling2D(pool_size = 2, strides = 2),
        tf.layers.Flatten(),
        tf.layers.Dense(units = 10)
        # tf.layers.Dense(units = 10, activation = 'softmax')
    ]

from functools import reduce
def sequential(layers, initial):
    return reduce(lambda l0,l1: l1(l0), layers, initial)

def composeModel(layers_fn, localization_fn, stn_placement, loop, shape):
    inp = tf.keras.layers.Input(shape=shape)
    layers = layers_fn()

    if localization_fn:
        stn = tf.keras.layers.Lambda(lambda inputs: transformer(inputs[0],inputs[1]))

        # first_layers = layers[:stn_placement]
        # localization_in = sequential(first_layers, inp)
        first_layers = tf.keras.layers.Lambda(lambda i: sequential(layers[:stn_placement], i))
        localization_in = first_layers(inp)

        parameters = tf.layers.Dense(
            units = 6,
            kernel_initializer = tf.keras.initializers.Zeros(),
            bias_initializer = tf.keras.initializers.Constant([1,0,0,1,0,0]),
        )(sequential(localization_fn(), localization_in))

        if loop:
            # first_out = sequential(first_layers, stn([inp, parameters]))
            first_out = first_layers(stn([inp, parameters]))
        else:
            first_out = stn([localization_in, parameters])

        pred = sequential(layers[stn_placement:], first_out)
        return tf.keras.models.Model(inputs=inp, outputs=pred)
    else:
        layers.insert(0, inp)
        return tf.keras.models.Sequential(layers)
        
