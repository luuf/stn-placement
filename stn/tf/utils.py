#%%
import tensorflow as tf
import random
import numpy as np
from transformer import spatial_transformer_network as transformer
import models
import data
from functools import reduce
try:
    from scipy.misc import imrotate
    imrotate_imported = True
except ImportError:
    print("Not importing imrotate")
    imrotate_imported = False

def rotate_tensor(im,rad=None):
    assert len(im.shape) == 4, "Im must have 4 dimensions to be rotated"
    B = tf.shape(im)[0]
    if rad is None:
        rad = tf.random_uniform([B],-np.pi/2,np.pi/2)
    c = tf.cos(rad)
    s = tf.sin(rad)
    zeros = tf.zeros([B])
    param = tf.stack([c, -s, zeros, s, c, zeros], 1) # pylint: disable=invalid-unary-operand-type
    return transformer(im, param)

def rotate_array(im,rad=None):
    assert im.ndim == 3, "Im must have 3 dimensions"
    assert imrotate_imported, "Can't use imrotate, scipy not available"
    if rad is None:
        deg = random.uniform(-90,90)
    else:
        deg = rad * 180 / np.pi
    return imrotate(im, deg, 'bilinear')

def concatenate_dictionaries(histories):
    return reduce(
        lambda d1,d2: {k: v + d2[k] for k,v in d1.items()},
        histories
    )
#%%
