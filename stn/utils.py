#%%
import tensorflow as tf
import numpy as np
from transformer import spatial_transformer_network as transformer

def rotaterandom(im,side,B):
    rad = tf.random_uniform([B],-np.pi/2,np.pi/2)
    c = tf.cos(rad)
    s = tf.sin(rad)
    zeros = tf.zeros([B])
    param = tf.stack([c, -s, zeros, s, c, zeros], 1)
    return tf.reshape(transformer(tf.reshape(im,[B,side,side,1]),param),(B,im.shape[1]))

def prepareslices(x,y,B,rotate=True):
    side = int(np.sqrt(x.shape[1])+0.5)
    x_slices = tf.data.Dataset.from_tensor_slices(x)
    y_slices = tf.data.Dataset.from_tensor_slices(y)
    x_slices = x_slices.repeat().batch(B)
    y_slices = y_slices.repeat().batch(B)
    if rotate:
        x_slices = x_slices.map(lambda im: rotaterandom(im,side,B))
    x_slices = x_slices.map(lambda x: tf.reshape(x,(B,side,side,1)))
    slices = tf.data.Dataset.zip((y_slices,x_slices))

    return slices

def prepareiterators(xtrn,ytrn,xval,yval,B):
    trn_slices = prepareslices(xtrn,ytrn,B)
    val_slices = prepareslices(xval,yval,B)
    handle = tf.placeholder(tf.string, shape=[], name='handle')
    handle_iterator = tf.data.Iterator.from_string_handle(handle,trn_slices.output_types,trn_slices.output_shapes)
    trn_iterator = trn_slices.make_one_shot_iterator()
    val_iterator = val_slices.make_one_shot_iterator()
    return (handle,handle_iterator,trn_iterator,val_iterator)

# def switchData(data_func):
#     global xtrn,xval,xtst,ytrn,yval,ytst,trnhandle,valhandle
#     xtrn,xval,xtst,ytrn,yval,ytst = data_func()
#     trn_flow = generator.flow(xtrn, ytrn, batch_size=B, shuffle=True)
#     tst_flow = generator.flow(xtst, ytst, batch_size=B, shuffle=True)

#%%
