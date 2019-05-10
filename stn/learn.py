#%% Setup
import tensorflow as tf
import numpy as np
from transformer import spatial_transformer_network as transformer
import time
import shelve
import dbm.dumb
import data
import models
from utils import * # pylint: disable=unused-wildcard-import
from argparse import ArgumentParser
from functools import reduce
from os import mkdir

print('Successful import')
#%% Command line arguments
parser = ArgumentParser()
parser.add_argument(
    "--name", '-n', type=str, 
    help="Name to save directory in"
)
parser.add_argument(
    "--dataset", '-d', type=str, 
    help="dataset to run on, default mnist"
)
parser.add_argument(
    "--iterations", '-i', type=int, 
    help="Iterations to train on, default 150000"
)
parser.add_argument(
    "--model", '-m', type=str, 
    help="Name of the model: CNN or FCN"
)
parser.add_argument(
    "--localization", '-l', type=str, 
    help="Name of localization: FCN, CNN, small or none"
)
parser.add_argument(
    "--stn-placement", '-p', type=int, 
    help="Number of layers to place stn after"
)
parser.add_argument(
    "--loop", type=bool, 
    help="Whether to place the stn in the beginning of the network or the middle"
)
parser.add_argument(
    "--runs", type=int, 
    help="Number of time to run this experiment, default 1"
)
parser.add_argument(
    "--rotate", type=bool, 
    help="Whether to rotate the dataset"
)
args = parser.parse_args()

print("Parsed: ", args)

#%% Set parameters
if args.dataset:
    data_fn = {
    # 'cluttered':    data.cluttered_mnist,
    # 'prerotated':   data.prerotated_mnist,
    # 'ownrotated':   data.ownrotated_mnist,
    'mnist':        data.mnist,
    'cifar10':      data.cifar10
    }.get(args.dataset)
    assert not (data_fn is None), 'Could not find dataset '+args.dataset
else:
    print('Using default dataset: mnist')
    data_fn = data.mnist

if args.model:
    layer_fn = {
        'CNN': models.CNN,
        'FCN': models.FCN
    }.get(args.model)
    assert not (layer_fn is None), 'Could not find model '+args.model
else:
    print('Using default model: CNN')
    layer_fn = models.CNN

if args.localization:
    localization_fn = {
        'CNN': models.CNN_localization,
        'FCN': models.FCN_localization,
        'small': models.small_localization,
        'none': False
    }.get(args.localization)
    assert not (localization_fn is None), 'Could not find localization '+args.localization
else:
    print('Using no spatial transformer network')
    localization_fn = False

stn_placement = args.stn_placement or 0
assert localization_fn or not stn_placement
it = args.iterations or 150000
loop = args.loop or False
runs = args.runs or 1
rotate = args.rotate or True
name = args.name or 'result'

#%% Setup model

xtrn,ytrn,xtst,ytst = data_fn()
samples = xtrn.shape[0]
B = 256 # batch size
B_per_epoch = np.floor(samples/B)

learning_rates = [0.01,0.001,0.0001]
switch_after_it = 50000
switch_after_epochs = int(B * switch_after_it / samples) # automatical floor
def scheduler(epoch):
    i = epoch // switch_after_epochs
    return learning_rates[i if len(learning_rates) > i else -1]
change_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

def save_all(directory, model, history, trn, tst, t):
    try:
        mkdir(directory)
    except FileExistsError:
        print('Overwriting existing directory:',name)
    file = open(directory + 'out.txt', 'w+')
    file.write('Training error: ' + str(trn) + '\n')
    file.write('Test error: ' + str(tst) + '\n')
    file.write('Time: ' + str(t) + '\n')

    model.save(directory + 'model.h5')
    db = dbm.dumb.open(directory + 'variables')
    with shelve.Shelf(db) as shelf:
        shelf['history'] = history.history
        shelf['samples'] = samples
        shelf['B'] = B
        shelf['args'] = args

def sequential(layers, initial):
    return reduce(lambda l0,l1: l1(l0), layers, initial)

layers = layer_fn()
inp = tf.keras.layers.Input(shape=xtrn.shape[1:])

for run in range(runs):
    if localization_fn:
        stn = tf.keras.layers.Lambda(lambda inputs: transformer(inputs[0],inputs[1]))

        first_layers = layers[:stn_placement]
        localization_in = sequential(first_layers, inp)
        # first_layers = tf.keras.layers.Lambda(lambda i: sequential(layers[:stn_placement], i))
        # localization_in = first_layers(inp)

        parameters = tf.layers.Dense(
            units = 6,
            kernel_initializer = tf.keras.initializers.Zeros(),
            bias_initializer = tf.keras.initializers.Constant([1,0,0,1,0,0]),
        )(sequential(localization_fn(), localization_in))

        if loop:
            first_out = sequential(first_layers, stn([inp, parameters]))
            # first_out = first_layers(stn([inp, parameters]))
        else:
            first_out = stn([localization_in, parameters])

        pred = sequential(layers[stn_placement:], first_out)
    else:
        pred = sequential(layers, inp)

    model = tf.keras.models.Model(inputs=inp, outputs=pred)

    optimizer = tf.keras.optimizers.SGD(lr=learning_rates[0])

    print("Type of labels", ytrn.dtype)

    model.compile(
        optimizer,
        loss = tf.losses.softmax_cross_entropy,
        metrics = ['accuracy'],
    )

    print("Compiled:", model)

#%% Train model

    if rotate:
        generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=90)
    else:
        generator = tf.keras.preprocessing.image.ImageDataGenerator()
    trn_flow = generator.flow(xtrn, ytrn, batch_size=B, shuffle=True)
    tst_flow = generator.flow(xtst, ytst, batch_size=B, shuffle=True)

    tf.keras.backend.get_session().run(tf.global_variables_initializer())

    t = time.time()
    epochs_to_train = int(it * B / samples)
    print('Training for epochs:', epochs_to_train)
    history = model.fit_generator(
        trn_flow, 
        epochs = epochs_to_train,
        # validation_data = tst_flow,
        callbacks = [change_lr]
    )
    steps_left = int(it - epochs_to_train * samples / B)
    print('Training for steps:', steps_left)
    for i,(x,y) in enumerate(trn_flow):
        if i >= steps_left:
            break
        train_batch = model.train_on_batch(x,y)
    else:
        raise Exception("Ran out of samples before steps")
    t = time.time() - t
    print('Time', t)
    print('Time per batch', t / it)

    print(history.history.keys())
    print('Evaluating')
    trn_res = model.evaluate_generator(trn_flow)
    print('Training accuracy:', trn_res)
    tst_res = model.evaluate_generator(tst_flow)
    print('Test accuracy:', tst_res)

    if runs == 1:
        save_all(name + '/', model, history, trn_res, tst_res, t)
    else:
        save_all(name + str(run) + '/', model, history, trn_res, tst_res, t)
