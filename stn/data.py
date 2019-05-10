#%%
import numpy as np
from tensorflow import keras

def mnist_reshape(x):
    return x.reshape(x.shape[0], 28, 28, 1)

def cluttered_mnist():
    data = np.load('../data/spatial-transformer-tensorflow-master/data/mnist_sequence1_sample_5distortions5x5.npz')
    xtrn = mnist_reshape(data['X_train'])
    xval = mnist_reshape(data['X_valid'])
    xtst = mnist_reshape(data['X_test'])
    ytrn = np.array([[float(y == i) for i in range(10)] for y in data['y_train']])
    yval = np.array([[float(y == i) for i in range(10)] for y in data['y_valid']])
    ytst = np.array([[float(y == i) for i in range(10)] for y in data['y_test']])
    return (xtrn,xval,xtst,ytrn,yval,ytst)

def prerotated_mnist():
    mtrn = np.load('../data/mnist_rotation_new/rotated_train.npz')
    mval = np.load('../data/mnist_rotation_new/rotated_valid.npz')
    mtst = np.load('../data/mnist_rotation_new/rotated_test.npz')

    xtrn = mnist_reshape(mtrn['x'])
    xval = mnist_reshape(mval['x'])
    xtst = mnist_reshape(mtst['x'])
    ytrn = np.array([[float(y == i) for i in range(10)] for y in mtrn['y']])
    yval = np.array([[float(y == i) for i in range(10)] for y in mval['y']])
    ytst = np.array([[float(y == i) for i in range(10)] for y in mtst['y']])

    return (xtrn,xval,xtst,ytrn,yval,ytst)

def ownrotated_mnist():
    data = np.load('../data/own_rot_mnist.npz')
    xtrn = mnist_reshape(np.reshape(data['x_train'][:50000],(50000,784)))
    xval = mnist_reshape(np.reshape(data['x_train'][50000:],(10000,784)))
    xtst = mnist_reshape(np.reshape(data['x_test'],(10000,784)))
    ytrn = np.array([[float(y == i) for i in range(10)] for y in data['y_train'][:50000]])
    yval = np.array([[float(y == i) for i in range(10)] for y in data['y_train'][50000:]])
    ytst = np.array([[float(y == i) for i in range(10)] for y in data['y_test']])
    return (xtrn,xval,xtst,ytrn,yval,ytst)

def mnist():
    (xtrn, ytrn), (xtst, ytst) = keras.datasets.mnist.load_data()
    xtrn = xtrn.reshape([xtrn.shape[0],28,28,1])
    xtst = xtst.reshape([xtst.shape[0],28,28,1])
    ytrn = np.array([[float(y == i) for i in range(10)] for y in ytrn])
    ytst = np.array([[float(y == i) for i in range(10)] for y in ytst])
    return (xtrn,ytrn,xtst,ytst)

def cifar10():
    (xtrn, ytrnind), (xtst, ytstind) = keras.datasets.cifar10.load_data()
    ytrn = np.array([[float(y == i) for i in range(10)] for y in ytrnind])
    ytst = np.array([[float(y == i) for i in range(10)] for y in ytstind])
    return (xtrn,ytrn,xtst,ytst)


#%%
