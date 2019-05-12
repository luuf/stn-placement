#%% Import
import tensorflow as tf
import tensorflow.keras as k
import numpy as np
from transformer import spatial_transformer_network as transformer
import time
import shelve
import dbm.dumb
import data
import models
import utils
from argparse import ArgumentParser
from os import mkdir

print('Successful import')
#%% Parse arguments
parser = ArgumentParser()
parser.add_argument(
    "--dataset", '-d', type=str, 
    help="dataset to run on, default mnist"
)
parser.add_argument(
    "--model", '-m', type=str, 
    help="Name of the model: CNN or FCN"
)
parser.add_argument(
    "--model-parameters", type=list,
    help="The number of neurons/filters to use in the model layers"
)
parser.add_argument(
    "--localization", '-l', type=str, 
    help="Name of localization: FCN, CNN, small or none"
)
parser.add_argument(
    "--localization-parameters", type=list,
    help="The number of neurons/filters to use in the localization layers"
)
parser.add_argument(
    "--stn-placement", '-p', type=int, 
    help="Number of layers to place stn after"
)
parser.add_argument(
    "--iterations", '-i', type=int, 
    help="Iterations to train on, default 150000"
)
parser.add_argument(
    "--runs", type=int, 
    help="Number of time to run this experiment, default 1"
)
parser.add_argument(
    "--name", '-n', type=str, 
    help="Name to save directory in"
)

loop_parser = parser.add_mutually_exclusive_group(required=False)
loop_parser.add_argument(
    "--loop", dest="loop", action="store_true",
    help="Use the stn-parameters to rotate the input, even if it's later"
)
loop_parser.add_argument(
    '--no-loop', dest='loop', action='store_false',
    help="Use the stn-parameters to rotate the featuremap it's placed after"
)
parser.set_defaults(loop=False)

rotate_parser = parser.add_mutually_exclusive_group(required=False)
rotate_parser.add_argument(
    "--rotate", dest="rotate", action="store_true",
    help="Rotate the data randomly before feeding it to the network"
)
rotate_parser.add_argument(
    '--no-rotate', dest='rotate', action='store_false',
    help="Use the data as-is"
)
parser.set_defaults(rotate=True)

args = parser.parse_args()
print("Parsed: ", args)

#%% Read arguments
if args.dataset is None:
    print('Using default dataset: mnist')
    args.dataset = 'mnist'
else:
    print('Using dataset:', args.dataset)
data_fn = data.data_dic.get(args.dataset)
assert not (data_fn is None), 'Could not find dataset'

if args.model is None:
    print('Using default model: CNN')
    args.model = 'CNN'
else:
    print('Using model:', args.model)
model_class = models.model_dic.get(args.model)
assert not (model_class is None), 'Could not find model'

model_obj = model_class(args.model_parameters)
if args.model_parameters is None:
    print('Using default parameters')
    args.model_parameters = model_obj.parameters

if args.localization is None:
    print('Using no spatial transformer network')
    args.localization = 'false'
else:
    print('Using localization:', args.localization)
localization_class = models.localization_dic.get(args.localization)
assert not (localization_class is None), 'Could not find localization'

localization_obj = localization_class(args.localization_parameters)
if localization_obj and args.localization_parameters is None:
    print('Using default parameters')
    args.localization_parameters = localization_obj.parameters

stn_placement = args.stn_placement or 0
print('STN-placement:', stn_placement)

it = args.iterations or 150000
print('Iterations:', it)

runs = args.runs or 1
print('Runs:', runs)

loop = args.loop
print('Loop:', bool(loop))

rotate = args.rotate
print('Rotate:', bool(rotate))

name = args.name or 'result'
print('Name:', name)

assert localization_class or (not stn_placement and not loop)

#%% Setup
xtrn,ytrn,xtst,ytst = data_fn()
samples = xtrn.shape[0]
B = 256 # batch size

learning_rates = [0.01,0.001,0.0001]
switch_after_it = 50000
switch_after_epochs = int(B * switch_after_it / samples) # automatical floor
eta = tf.Variable(learning_rates[0],trainable=False)
def change_learningrate(epoch,logs):
    print('learning rate epoch', epoch)
    if epoch % switch_after_epochs == 0:
        i = epoch // switch_after_epochs
        print('i',i)
        eta.assign(learning_rates[i if i < len(learning_rates) else -1])
callback = k.callbacks.LambdaCallback(on_epoch_begin=change_learningrate)

for run in range(runs):
    # Create model
    classification_model = models.compose_model(model_obj, localization_obj, stn_placement, loop, xtrn.shape[1:])

    if rotate:
        model = models.add_rotation_layer(classification_model)
    else:
        model = classification_model

    model.compile(
        tf.train.GradientDescentOptimizer(eta),
        # k.optimizers.SGD(lr=learning_rates[0]),
        loss = tf.losses.softmax_cross_entropy,
        metrics = ['accuracy'],
    )
    print("Compiled:", model)

    k.backend.get_session().run(tf.global_variables_initializer())

    # Train model
    epochs_to_train = int(it * B / samples)
    print('Training for epochs:', epochs_to_train)
    t = time.time()
    history = model.fit(
        x = xtrn,
        y = ytrn,
        batch_size = 256,
        epochs = epochs_to_train,
        shuffle = True,
        callbacks = [callback]
        # validation_data = tst_flow,
    )
    steps_left = int(it - epochs_to_train * samples / B)
    print('Training for steps:', steps_left)
    model.fit(
        x = xtrn[:B*steps_left],
        y = ytrn[:B*steps_left],
        batch_size = 256,
        epochs = 1,
        initial_epoch = epochs_to_train,
        shuffle = True
        # callbacks = [change_lr]
    )
    t = time.time() - t
    print('Time', t)
    print('Time per batch', t / it)

    # Evaluate model
    print('Evaluating...')
    trn_res = model.evaluate(x=xtrn,y=ytrn,batch_size=256)
    print('Training accuracy:', trn_res)
    tst_res = model.evaluate(x=xtst,y=ytst,batch_size=256)
    print('Test accuracy:', tst_res)

    print('Keys saved:', history.history.keys())

    # Save run
    directory = (name if runs == 1 else name + str(run)) + '/'

    try:
        mkdir(directory)
    except FileExistsError:
        print('Overwriting existing directory:',name)

    file = open(directory + 'out.txt', 'w+')
    file.write('Training error: ' + str(trn_res) + '\n')
    file.write('Test error: ' + str(tst_res) + '\n')
    file.write('Time: ' + str(t) + '\n')

    classification_model.save(
        directory + 'model.h5',
        include_optimizer = False
    )

    db = dbm.dumb.open(directory + 'variables')
    with shelve.Shelf(db) as shelf:
        shelf['history'] = history.history
        shelf['samples'] = samples
        shelf['B'] = B

        shelf['dataset'] = args.dataset
        shelf['model'] = args.model
        shelf['model_parameters'] = args.model_parameters
        shelf['localization'] = args.localization
        shelf['localization_parameters'] = args.localization_parameters
        shelf['stn_placement'] = stn_placement
        shelf['loop'] = loop
        shelf['iterations'] = it
        shelf['rotate'] = rotate