#%% Import
import torch as t
import torchvision as tv
import numpy as np
import time
from argparse import ArgumentParser
from os import makedirs
import data
import models
from datetime import datetime

print('Launched at', datetime.now())
#%% Parse arguments
parser = ArgumentParser()
parser.add_argument(
    "--dataset", '-d', type=str, default='mnist',
    help="dataset to run on, default mnist"
)
parser.add_argument(
    "--model", '-m', type=str, default='CNN',
    help="Name of the model: CNN or FCN"
)
parser.add_argument(
    "--model-parameters", nargs="*", type=int,
    help="The number of neurons/filters to use in the model layers"
)
parser.add_argument(
    "--localization", '-l', type=str, default='false',
    help="Name of localization: FCN, CNN, small or none"
)
parser.add_argument(
    "--localization-parameters", nargs="*", type=int,
    help="The number of neurons/filters to use in the localization layers"
)
parser.add_argument(
    "--stn-placement", '-p', type=int, default=0,
    help="Number of layers to place stn after"
)
parser.add_argument(
    "--epochs", '-e', type=int, default=640,
    help="Epochs to train on, default 640" # 640 = 150000 * 256 / 60000
)
parser.add_argument(
    "--runs", type=int, default=1,
    help="Number of time to run this experiment, default 1"
)
parser.add_argument(
    "--name", '-n', type=str, default='result',
    help="Name to save directory in"
)
parser.add_argument(
    "--optimizer", '-o', type=str, default='sgd',
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
data_fn = data.data_dict.get(args.dataset)
if data_fn is None:
    print('Using precomputed dataset',args.dataset)
    assert args.rotate is False
    train_loader, test_loader = data.get_precomputed(args.dataset)
else:
    print('Using dataset:', args.dataset, '; rotated' if args.rotate else '')
    train_loader, test_loader = data_fn(args.rotate)
input_shape = train_loader.dataset[0][0].shape


print('Using model:', args.model)
model_class = models.model_dict.get(args.model)
assert not (model_class is None), 'Could not find model'

model_obj = model_class(args.model_parameters)
if args.model_parameters is None or args.model_parameters == []:
    print('Using default parameters')
    args.model_parameters = model_obj.param


print('Using localization:', args.localization)
localization_class = models.localization_dict.get(args.localization)
assert not (localization_class is None), 'Could not find localization'

localization_obj = localization_class(args.localization_parameters)
no_parameters = args.localization_parameters is None or args.localization_parameters == []
assert localization_obj or no_parameters, "localization parameters can't be used without stn"
if localization_obj and no_parameters:
    print('Using default parameters')
    args.localization_parameters = localization_obj.param


print('Using optimizer',args.optimizer)
optimizer_fn = {
    'sgd': t.optim.SGD,
    'adam': t.optim.Adam
}.get(args.optimizer)
assert not optimizer_fn is None, 'Could not find optimizer'

assert localization_class or (not args.stn_placement and not args.loop)

#%% Setup
if args.lr is None:
    learning_rate = 0.01
    learning_rate_multipliers = [1,0.1,0.01]
    switch_after_epochs = 214 # 50000 * 256 / 60000
else:
    learning_rate = args.lr
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

directory = args.name + '/'
makedirs(directory, exist_ok=True)

#%% Run
for run in range(args.runs):

    prefix = str(run) if args.runs > 1 else ''

    history = {
        'train_loss': np.zeros(args.epochs,),
        'test_loss': np.zeros(args.epochs,),
        'train_acc': np.zeros(args.epochs,),
        'test_acc': np.zeros(args.epochs,)
    }

    # Create model
    model = models.Net(model_obj, localization_obj, args.stn_placement, args.loop, input_shape)
    model = model.to(device)

    # initialize

    # Train model
    print('Training for epochs:', args.epochs)
    print('Switching learning rate after', switch_after_epochs)
    start_time = time.time()

    optimizer = optimizer_fn(model.parameters(), learning_rate)
    scheduler = t.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: learning_rate_multipliers[e // switch_after_epochs]
    )

    for epoch in range(args.epochs):
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
    print('Time per epoch', total_time / args.epochs)

    print('Train accuracy:', history['train_acc'][-1])
    print('Test accuracy:', history['test_acc'][-1])
    print()
    final_accuracies['train'].append(history['train_acc'][-1])
    final_accuracies['test'].append(history['test_acc'][-1])

    t.save(model.state_dict(), directory + prefix + 'final')
    t.save(history, directory + prefix + 'history')

for run in range(args.runs):
    print('Train accuracy:', final_accuracies['train'][run])
    print('Test accuracy:', final_accuracies['test'][run])
# Save model details
t.save(
    {
        'dataset': args.dataset,
        'rotate': args.rotate,
        'model': args.model,
        'model_parameters': args.model_parameters,
        'localization': args.localization,
        'localization_parameters': args.localization_parameters,
        'stn_placement': args.stn_placement,
        'loop': args.loop,
        'learning_rate': learning_rate,
        'epochs': args.epochs,
    },
    directory + 'model_details',
)
