#%%
from scipy.misc import imrotate as rot
import numpy as np
import random

#%% Get mnist
import mnist

mnist.init()
x_train, t_train, x_test, t_test = mnist.load()

#%% generate rotaded mnist
# def rotaterandom(arr):
    # return rot(np.reshape(arr,(28,28)),random.uniform(-90,90),'bilinear')
func = lambda x: rot(np.reshape(x,(28,28)),random.uniform(-90,90),'bilinear').astype('float32')/255
trn_rotated = list(map(func, x_train))
tst_rotated = list(map(func, x_test))

np.savez("own_rot_mnist",x_train=trn_rotated,y_train=t_train,x_test=tst_rotated,y_test=t_test)

#%% generate unrotated mnist
x_trn = (x_train/255).astype('float32')
x_tst = (x_test/255).astype('float32')
np.savez("mnist",x_train=x_trn,y_train=t_train,x_test=x_tst,y_test=t_test)
