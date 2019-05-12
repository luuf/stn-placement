#%% Import modules
import tensorflow as tf
import numpy as np
from transformer import spatial_transformer_network as transformer
import matplotlib.pyplot as plt
import time
from scipy.misc import imrotate
import shelve
import data
import models

#%% Restore data
# directory = 'result'
directory = '../result'
# directory = '../../legacy/CNNSTNmp'
with shelve.open(directory + '/variables', flag='r') as shelf:
    try:
        history = shelf['history']
        samples = shelf['samples']
        B = shelf['B']

        data_fn = data.data_dic.get(shelf['dataset'])
        xtrn, ytrn, xtst, ytst = data_fn()

        # model_class = models.model_dic.get(shelf['model'])
        # model_obj = model_class(shelf['model_parameters'])

        # localization_class = models.model_dic.get(shelf['localization'])
        # localization_obj = localization_class(shelf['localization_parameters'])

        # it = shelf['iterations']
        # rotate = shelf['rotate'] 

        # model = models.compose_model(
        #     model_obj,
        #     localization_obj,
        #     shelf['stn_placement'],
        #     shelf['loop'],
        #     xtrn.shape[1:]
        # )
    except KeyError:
        print('Old data file, restoring everything that can be restored')
        for key in shelf:
            globals()[key] = shelf[key]

with tf.keras.utils.CustomObjectScope({'STN':models.STN}):
    model = tf.keras.models.load_model(
        filepath = directory + '/model.h5',
        custom_objects = {
            'softmax_cross_entropy':tf.losses.softmax_cross_entropy,
            'transformer': transformer
        }
    )

#%%
# plt.plot(trn_val)
# plt.plot(acc_val) 
plt.plot(history['acc'])
plt.figure()
# plt.plot(history['val_acc'])
plt.show()
#%% functions
def plotarray(arr):
    side = int(np.sqrt(np.size(arr)) + 0.5)
    plt.imshow(np.reshape(arr,(side,side)))
    plt.show()

def plotfirst(mat):
    plotarray(mat[0])

def plottransformed(arr,param):
    side = int(np.sqrt(np.size(arr)) + 0.5)
    plt.imshow(np.reshape(arr,(side,side)))
    plt.figure()
    toplot = transformer(np.reshape(arr,(1,side,side,1)),param)
    with tf.Session() as sess:
        plt.imshow(np.reshape(sess.run(toplot),(side,side)))
    plt.show()

def rotatearray(arr,deg,plot=True):
    side = int(np.sqrt(np.size(arr)) + 0.5)
    square = np.reshape(arr,(side,side))
    rotated = imrotate(square,deg,'bilinear')
    if plot:
        plt.imshow(square)
        plt.figure()
        plt.imshow(rotated)
        plt.show()
    return np.reshape(rotated/255,arr.shape).astype('float32')

def finalTrans(n=10):
    for i in range(n):
        plotarray(inp_arr[-1][i])
        plotarray(rot_arr[-1][i])
    
# legacy function
def finalParameters(n=10):
    with tf.Session() as sess:
        transformed = sess.run(transformer(inp_arr[-1][0:n],rot_arr[-1][0:n]))
    for i in range(n):
        plt.imshow(inp_arr[-1][i,:,:,0])
        plt.figure()
        plt.imshow(transformed[i,:,:,0])
        plt.show()

def compareFinalTransWith(epochs): # earlier
    n = 10
    first_set = -round(epochs * samples/B) - 1
    snd_set = -1
    for f,s in zip(rot_arr[first_set][:n],rot_arr[snd_set][:n]):
        plotarray(f)
        plotarray(s)

def switchData(data_func):
    global xtrn,xval,xtst,ytrn,yval,ytst,trninit,valinit
    xtrn,xval,xtst,ytrn,yval,ytst = data_func()
    _,trninit = preparedata(xtrn,ytrn,'CNN',iterator)
    _,valinit = preparedata(xval,yval,'CNN',iterator)

def rotBefAftConv(deg):
    outp,inp,conv = sess.run((nxt[0],nxt[1],h_trans))
    rotinp = list(map(lambda x: np.reshape(rotatearray(x,deg,plot=False),(784,)), inp))
    _,rotinit = preparedata(np.array(rotinp),outp,'CNN',iterator)
    sess.run(rotinit)
    out2 = sess.run((nxt[1],h_trans))

    for i in range(5):
        plotarray(inp[i])
        plotarray(rotatearray(conv[i][:,:,0],deg,plot=False))
        plotarray(rotatearray(conv[i][:,:,1],deg,plot=False))
        plotarray(rotinp[i])
        plotarray(out2[1][i][:,:,0])
        plotarray(out2[1][i][:,:,1])

def plotTransformedMiddle(n = 10):
    for i in range(n):
        plt.imshow(internal_arr[-1][0][i,:,:,0])
        plt.figure()
        plt.imshow(internal_arr[-1][1][i,:,:,0])
        plt.show()

def runTransformedMiddle(n = 10):
    B = 256
    xtrn,ytrn,xval,yval,xtst,ytst = mnist()
    handle = tf.get_default_graph().get_tensor_by_name('handle:0')
    _,iterator,trn_it,val_it = prepareiterators(xtrn,ytrn,xval,yval,B)
    transformed = tf.get_default_graph().get_tensor_by_name('transformed:0')
    trnhandle = sess.run(trn_it.string_handle())
    res = sess.run(transformed, feed_dict={handle:trnhandle})
    for i in range(10):
        plt.plot(res[i])
        plt.show()


#%%