#%% Setup
import tensorflow as tf
import numpy as np
from stn import spatial_transformer_network as transformer
import time
import shelve
import dbm.dumb
from stn_utils import * # pylint: disable=unused-wildcard-import

# xtrn,xval,xtst,ytrn,yval,ytst = cluttered_mnist()
xtrn,xval,xtst,ytrn,yval,ytst = mnist()

#%% Tensorflow
B = 256 # batch size
it = 150000
samples,size = xtrn.shape
B_per_epoch = np.floor(samples/B)
side = int(np.sqrt(size)+0.5)
activation_f = tf.nn.relu

# identity transform
n_fc = 6
initial = np.array([[1., 0, 0], [0, 1., 0]])
initial = initial.astype('float32').flatten()

#% Architecture
def small_localization(inp):
    loc_dense = tf.layers.Dense(units = 32, activation = activation_f)
    W = tf.Variable(tf.zeros([32, n_fc]), name='W_fc1')
    b = tf.Variable(initial_value=initial, name='b_fc1')
    reshaped = tf.reshape(inp,(B,tf.reduce_prod(inp.shape[1:])))
    return tf.add(tf.matmul(loc_dense(reshaped), W), b, name='parameters')

def FCN_localization(inp):
    loc_dense1 = tf.layers.Dense(units = 32, activation = activation_f)
    loc_dense2 = tf.layers.Dense(units = 32, activation = activation_f)
    loc_dense3 = tf.layers.Dense(units = 32, activation = activation_f)
    W = tf.Variable(tf.zeros([32, n_fc]), name='W_fc1')
    b = tf.Variable(initial_value=initial, name='b_fc1')
    reshaped = tf.reshape(inp,(B,tf.reduce_prod(inp.shape[1:])))
    return tf.add(tf.matmul(loc_dense3(loc_dense2(loc_dense1(reshaped))), W), b, name='parameters')

def CNN_localization(inp,downsampled=True):
    loc_conv1 = tf.layers.Conv2D(filters=20, kernel_size=(5,5), activation = activation_f)
    loc_mp = tf.layers.MaxPooling2D(pool_size = 2, strides = 2)
    loc_conv2 = tf.layers.Conv2D(filters=20, kernel_size=(5,5), activation = activation_f)
    # mp_2 = tf.layers.MaxPooling2D(pool_size = 2, strides = 2)
    loc_dense1 = tf.layers.Dense(units = 20, activation = activation_f)
    W = tf.Variable(tf.zeros([20, n_fc]), name='W_fc1')
    b = tf.Variable(initial_value=initial, name='b_fc1')

    if downsampled:
        halfside = round(side/2)
        downsampled = tf.image.resize_images(inp,(halfside,halfside))
        conv = loc_conv2(loc_mp(loc_conv1(downsampled)))
        final_size = round((halfside - 4)/2 - 4)**2 * 20
    else:
        conv = loc_conv2(loc_mp(loc_conv1(inp)))
        final_size = round((int(inp.shape[1]) - 4)/2 - 4)**2 * 20

    return tf.add(tf.matmul(loc_dense1(tf.reshape(conv,[B,final_size])), W), b, name='parameters')

def STNFCN(inp):
    parameters = FCN_localization(inp)

    # spatial transformer layer
    transformed = transformer(inp, parameters)
    transformed = tf.reshape(transformed,(B,size),name='transformed')

    # other layers
    dense1 = tf.layers.Dense(units = 256, activation = activation_f)
    dense2 = tf.layers.Dense(units = 200, activation = activation_f)
    out = tf.layers.Dense(units = 10)

    pred = out(dense2(dense1(transformed)))

    return (pred,(transformed,))

def STNCNN(inp):
    parameters = CNN_localization(inp)

    # spatial transformer layer
    transformed = transformer(inp, parameters) # 1 is reshaped
    transformed = tf.identity(transformed,name='transformed')

    # other layers
    conv1 = tf.layers.Conv2D(filters = 64, kernel_size = (9,9), activation = activation_f)
    mp1 = tf.layers.MaxPooling2D(pool_size = 2, strides = 2)
    conv2 = tf.layers.Conv2D(filters = 64, kernel_size = (7,7), activation = activation_f)
    mp2 = tf.layers.MaxPooling2D(pool_size = 2, strides = 2)
    out = tf.layers.Dense(units = 10)

    conv = mp2(conv2(mp1(conv1(transformed))))
    final_size = (((int(inp.shape[1]) - 8)//2 - 6)//2)**2 * 64
    pred = out(tf.reshape(conv,[B,final_size]))

    return (pred,(transformed,))

# Network with a STN partway through it's network
def CNNSTN(inp):
    # other layers
    conv1 = tf.layers.Conv2D(filters = 64, kernel_size = (9,9), activation = activation_f)
    mp1 = tf.layers.MaxPooling2D(pool_size = 2, strides = 2)
    conv2 = tf.layers.Conv2D(filters = 64, kernel_size = (7,7), activation = activation_f)
    mp2 = tf.layers.MaxPooling2D(pool_size = 2, strides = 2)
    out = tf.layers.Dense(units = 10)

    fstconv = mp1(conv1(inp))
    parameters = FCN_localization(fstconv)

    transformed = transformer(fstconv, parameters)
    transformed = tf.identity(transformed, name='transformed')
    conv = mp2(conv2(transformed))
    #transformed = transformer(inp,parameters)
    #transformed = tf.identity(transformed, name='transformed')
    #conv = mp2(conv2(mp1(conv1(transformed))))

    final_size = (((side - 8)//2 - 6)//2)**2 * 64
    pred = out(tf.reshape(conv,[B,final_size]))

    return (pred,(parameters,fstconv,transformed))

def CNN(inp):
    conv1 = tf.layers.Conv2D(filters = 64, kernel_size = (9,9), activation = activation_f)
    mp1 = tf.layers.MaxPooling2D(pool_size = 2, strides = 2)
    conv2 = tf.layers.Conv2D(filters = 64, kernel_size = (7,7), activation = activation_f)
    mp2 = tf.layers.MaxPooling2D(pool_size = 2, strides = 2)
    out = tf.layers.Dense(units = 10)

    conv = mp2(conv2(mp1(conv1(inp))))
    final_size = (((side - 8)//2 - 6)//2)**2 * 64
    pred = out(tf.reshape(conv,[B,final_size]))

    return (pred,(conv1,))

tf.reset_default_graph()
handle,iterator,trn_it,val_it = prepareiterators(xtrn,ytrn,xval,yval,B)
nxt = iterator.get_next()
pred,internals = CNNSTN(nxt[1])

#%% Optimizer and metrics
loss = tf.losses.softmax_cross_entropy(onehot_labels = tf.reshape(nxt[0],[B,10]), logits = pred)
eta = tf.Variable(0.01,trainable=False)
train = tf.train.GradientDescentOptimizer(eta).minimize(loss)

pred_num = tf.argmax(pred,1)
y_num = tf.argmax(nxt[0],1)
_,c_accuracy = tf.metrics.accuracy(labels = y_num, predictions = pred_num, name='c_accuracy')
reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

#%% Initialize
runopt = tf.RunOptions(report_tensor_allocations_upon_oom = True)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

trnhandle = sess.run(trn_it.string_handle())
valhandle = sess.run(val_it.string_handle())

acc_trn = []
acc_val = []
h_arr = []
inp_arr = []
rot_arr = []
if len(internals) > 1:
    more_internals = True
    internal_arr = []
else:
    more_internals = False

#%% Run graph
learning_rates = [0.01,0.001,0.0001]
t = time.time()
for lr in learning_rates:
    eta.assign(lr)
    for i in range(round(it/len(learning_rates))):
        if i % B_per_epoch == 0:
            if i % (B_per_epoch * 10) == 0:
                out = sess.run((train,c_accuracy,nxt[1])+internals,options=runopt,feed_dict={handle: trnhandle})
                inp_arr.append(out[2])
                rot_arr.append(out[3])
                if more_internals:
                    internal_arr.append(out[4:])
            else:
                out = sess.run((train,c_accuracy),options=runopt,feed_dict={handle: trnhandle})
            acc_trn.append(out[1])
            sess.run(reset_metrics)
            acc_val.append(sess.run(c_accuracy,feed_dict={handle: valhandle}))
            sess.run(reset_metrics)
        else:
            sess.run(train,options=runopt,feed_dict={handle: trnhandle})
tot = time.time() - t
print(tot, tot/it)

for i in range(100):
    out = sess.run(c_accuracy,feed_dict={handle: trnhandle})
sess.run(reset_metrics)
for i in range(int(yval.shape[0]/B)):
    valout = sess.run(c_accuracy,feed_dict={handle: valhandle})
print('Trn: ' + str(out))
print('Val: ' + str(valout))

#%% More functions
def tenboard(dir='.'):
    writer = tf.summary.FileWriter(dir)
    writer.add_graph(tf.get_default_graph())
    writer.flush()

def switchData(data_func):
    global xtrn,xval,xtst,ytrn,yval,ytst,trnhandle,valhandle
    xtrn,xval,xtst,ytrn,yval,ytst = data_func()
    trnhandle = sess.run(prepareslices(xtrn,ytrn,B).make_one_shot_iterator().string_handle())
    valhandle = sess.run(prepareslices(xtrn,ytrn,B).make_one_shot_iterator().string_handle())

def saveModel():
    tf.train.Saver().save(sess,'savedModels/rotmnist50000itCNN')

#%% Save data
directory = 'result/'
tf.train.Saver().save(sess,directory+'tfsession')
sess.close()
db = dbm.dumb.open(directory+'variables')
shelf = shelve.Shelf(db)
shelf['acc_trn'] = acc_trn
shelf['acc_val'] = acc_val
shelf['inp_arr'] = inp_arr
shelf['rot_arr'] = rot_arr
shelf['samples'] = samples
shelf['B'] = B
if more_internals:
    shelf['internal_arr'] = internal_arr
shelf.close()

#%%
