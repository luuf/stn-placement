#%%
import tensorflow as tf
import numpy as np
from transformer import spatial_transformer_network as transformer
import models
import data

def rotate_with_stn(im,rad=None):
    assert tf.rank(im) == 4, "Im must be of rank 4 to be rotated"
    B = tf.shape(im)[0]
    if rad is None:
        rad = tf.random_uniform([B],-np.pi/2,np.pi/2)
    c = tf.cos(rad)
    s = tf.sin(rad)
    zeros = tf.zeros([B])
    param = tf.stack([c, -s, zeros, s, c, zeros], 1) # pylint: disable=invalid-unary-operand-type
    return transformer(im, param)


#%%
