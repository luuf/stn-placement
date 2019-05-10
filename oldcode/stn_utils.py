import tensorflow as tf
import numpy as np
from stn import spatial_transformer_network as transformer

def cluttered_mnist():
    data = np.load('../data/spatial-transformer-tensorflow-master/data/mnist_sequence1_sample_5distortions5x5.npz')
    xtrn = data['X_train']
    xval = data['X_valid']
    xtst = data['X_test']
    ytrn = np.array([[float(y == i) for i in range(10)] for y in data['y_train']])
    yval = np.array([[float(y == i) for i in range(10)] for y in data['y_valid']])
    ytst = np.array([[float(y == i) for i in range(10)] for y in data['y_test']])
    return (xtrn,xval,xtst,ytrn,yval,ytst)

def rotated_mnist():
    mtrn = np.load('../data/mnist_rotation_new/rotated_train.npz')
    mval = np.load('../data/mnist_rotation_new/rotated_valid.npz')
    mtst = np.load('../data/mnist_rotation_new/rotated_test.npz')

    xtrn = mtrn['x']
    xval = mval['x']
    xtst = mtst['x']
    ytrn = np.array([[float(y == i) for i in range(10)] for y in mtrn['y']])
    yval = np.array([[float(y == i) for i in range(10)] for y in mval['y']])
    ytst = np.array([[float(y == i) for i in range(10)] for y in mtst['y']])

    return (xtrn,xval,xtst,ytrn,yval,ytst)

def own_rot_mnist():
    data = np.load('../data/own_rot_mnist.npz')
    xtrn = np.reshape(data['x_train'][:50000],(50000,784))
    xval = np.reshape(data['x_train'][50000:],(10000,784))
    xtst = np.reshape(data['x_test'],(10000,784))
    ytrn = np.array([[float(y == i) for i in range(10)] for y in data['y_train'][:50000]])
    yval = np.array([[float(y == i) for i in range(10)] for y in data['y_train'][50000:]])
    ytst = np.array([[float(y == i) for i in range(10)] for y in data['y_test']])
    return (xtrn,xval,xtst,ytrn,yval,ytst)

def mnist():
    data = np.load('../data/mnist.npz')
    xtrn = np.reshape(data['x_train'][:50000],(50000,784))
    xval = np.reshape(data['x_train'][50000:],(10000,784))
    xtst = np.reshape(data['x_test'],(10000,784))
    ytrn = np.array([[float(y == i) for i in range(10)] for y in data['y_train'][:50000]])
    yval = np.array([[float(y == i) for i in range(10)] for y in data['y_train'][50000:]])
    ytst = np.array([[float(y == i) for i in range(10)] for y in data['y_test']])
    return (xtrn,xval,xtst,ytrn,yval,ytst)

def rotaterandom(im,side,B):
    rad = tf.random_uniform([B],-np.pi/2,np.pi/2)
    c = tf.cos(rad)
    s = tf.sin(rad)
    zeros = tf.zeros([B])
    param = tf.stack([c,-s,zeros,s,c,zeros],1)
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