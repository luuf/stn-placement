#%% Import
import torch as t
import torchvision as tv
import numpy as np
import time
import datatorch as data
import modelstorch as models
# import utils
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
    "--model-parameters", nargs="*", type=int,
    help="The number of neurons/filters to use in the model layers"
)
parser.add_argument(
    "--localization", '-l', type=str, 
    help="Name of localization: FCN, CNN, small or none"
)
parser.add_argument(
    "--localization-parameters", nargs="*", type=int,
    help="The number of neurons/filters to use in the localization layers"
)
parser.add_argument(
    "--stn-placement", '-p', type=int, 
    help="Number of layers to place stn after"
)
parser.add_argument(
    "--epochs", '-e', type=int, 
    help="Epochs to train on, default 640" # 640 = 150000 * 256 / 60000
)
parser.add_argument(
    "--runs", type=int, 
    help="Number of time to run this experiment, default 1"
)
parser.add_argument(
    "--name", '-n', type=str, 
    help="Name to save directory in"
)
parser.add_argument(
    "--optimizer", '-o', type=str,
    help="Name of the optimizer"
)
parser.add_argument(
    "--lr", type=float,
    help="Constant learning rate to use. Default is 0.01, 0.001 0.0001"
)
# parser.add_argument(
#     "--dropout", type=float,
#     help="Fraction of units to drop. Default is to not use dropout at all."
# )

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
parser.set_defaults(rotate=False)

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

# dropout = args.dropout
# if dropout is not None:
#     print('Using dropout', dropout)

if args.model is None:
    print('Using default model: CNN')
    args.model = 'CNN'
else:
    print('Using model:', args.model)
model_class = models.model_dic.get(args.model)
assert not (model_class is None), 'Could not find model'

model_obj = model_class(args.model_parameters)
if args.model_parameters is None or args.model_parameters == []:
    print('Using default parameters')
    args.model_parameters = model_obj.param

if args.localization is None:
    print('Using no spatial transformer network')
    args.localization = 'false'
else:
    print('Using localization:', args.localization)
localization_class = models.localization_dic.get(args.localization)
assert not (localization_class is None), 'Could not find localization'

localization_obj = localization_class(args.localization_parameters)
no_parameters = args.localization_parameters is None or args.localization_parameters == []
assert localization_obj or no_parameters
if localization_obj and no_parameters:
    print('Using default parameters')
    args.localization_parameters = localization_obj.param

stn_placement = args.stn_placement or 0
print('STN-placement:', stn_placement)

epochs = args.epochs or 640
print('Epochs:', epochs)

runs = args.runs or 1
print('Runs:', runs)

loop = args.loop
print('Loop:', bool(loop))

rotate = args.rotate
print('Rotate:', bool(rotate))

name = args.name or 'result'
print('Name:', name)

if args.optimizer is None:
    print('Using default optimizer GradientDescentOptimizer')
    args.optimizer = 'sgd'
else:
    print('Using optimizer',args.optimizer)
optimizer_fn = {
    'sgd': t.optim.SGD,
    'adam': t.optim.Adam
}.get(args.optimizer)
assert not optimizer_fn is None, 'Could not find optimizer'

learning_rate = args.lr
print("Learning_rate", learning_rate or "default variable")

assert localization_class or (not stn_placement and not loop)

#%% Setup
train_loader, test_loader = data_fn(rotate)
input_shape = train_loader.dataset[0][0].shape

if learning_rate is None:
    learning_rate = 0.01
    learning_rate_multipliers = [1,0.1,0.01]
    switch_after_epochs = 214 # 50000 * 256 / 60000
else:
    learning_rate_multipliers = [1]
    switch_after_epochs = np.inf

device = t.device("cuda" if t.cuda.is_available() else "cpu") # pylint: disable=no-member

cross_entropy = t.nn.CrossEntropyLoss()

final_accuracies = {'train':[], 'test':[]}
# These need to be defined before train and test
optimizer = None
scheduler = None
history = None

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) 
        optimizer.zero_grad()
        output = model(data)
        loss = cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        batch_size = data.shape[0]
        history['train_loss'][epoch] += loss.item() * batch_size
        pred = output.max(1, keepdim=True)[1]
        history['train_acc'][epoch] += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % 50 == 0 and device == t.device("cpu"): # pylint: disable=no-member
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    history['train_loss'][epoch] /= len(train_loader.dataset)
    history['train_acc'][epoch] /= len(train_loader.dataset)

def test(epoch = None):
    cross_entropy_sum = t.nn.CrossEntropyLoss(reduction='sum')
    with t.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += cross_entropy_sum(output, target).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)

    if not epoch is None:
        history['test_loss'][epoch] = test_loss
        history['test_acc'][epoch] = correct
    return test_loss, correct


directory = 'experiments/' + name + '/'

try:
    mkdir(directory)
    print('Creating directory', directory)
except FileExistsError:
    print('Overwriting existing directory', directory)

for run in range(runs):

    prefix = str(run) if runs > 1 else ''

    history = {
        'train_loss': np.zeros(epochs,),
        'test_loss': np.zeros(epochs,),
        'train_acc': np.zeros(epochs,),
        'test_acc': np.zeros(epochs,)
    }

    # Create model
    model = models.Net(model_obj, localization_obj, stn_placement, loop, input_shape)
    model = model.to(device)

    # initialize

    # Train model
    print('Training for epochs:', epochs)
    print('Switching learning rate after', switch_after_epochs)
    start_time = time.time()

    optimizer = optimizer_fn(model.parameters(), learning_rate)
    scheduler = t.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: learning_rate_multipliers[e // switch_after_epochs]
    )

    for epoch in range(epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
        if epoch % 100 == 0:
            # TODO: ADD SAVING OF OPTIMIZER AND OTHER POTENTIALLY RELEVANT THINGS
            t.save(model.state_dict(), directory + prefix + 'ckpt' + str(epoch))
            print(
                'Saved model at epoch', epoch, '\n'
                'Train',history['train_acc'][epoch],
                'Test', history['test_acc'][epoch]
            )
    total_time = time.time() - start_time
    print('Time', total_time)
    print('Time per epoch', total_time / epochs)

    print('Train accuracy:', history['train_acc'][-1])
    print('Test accuracy:', history['test_acc'][-1])
    print()
    final_accuracies['train'][run] = ['train_acc'][-1]
    final_accuracies['test'][run] = ['test_acc'][-1]

    t.save(model.state_dict(), directory + prefix + 'final')
    t.save(history, directory + prefix + 'history')

for run in runs:
    print('Train accuracy:', final_accuracies['train'][run])
    print('Test accuracy:', final_accuracies['test'][run])
# Save model details
t.save({
    'dataset': args.dataset,
    'rotate': rotate,
    'model': args.model,
    'model_parameters': args.model_parameters,
    'localization': args.localization,
    'localization_parameters': args.localization_parameters,
    'stn_placement': stn_placement,
    'loop': loop,
    'learning_rate': learning_rate,
    'epochs': epochs,
}, directory + 'model_details')

