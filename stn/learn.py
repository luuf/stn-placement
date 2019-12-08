#%% Import
import torch as t
import torchvision as tv
import numpy as np
import time
import eval
from argparse import ArgumentParser
from os import makedirs, path
import data
import models
from datetime import datetime
from functools import partial
from ylvas_code.lrdecay_functions import StepLRBase

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
    "--stn-placement", '-p', nargs="*", type=int, default=[],
    help="Number of layers to place stn after"
)
# parser.add_argument(
#     "--batchnorm-placement", nargs="*", type=int, default=[],
#     help="Indices of layers to place batch normalization before"
# )
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
    "--lr", type=float, default=0.01,
    help="Constant learning rate to use. Default is 0.01"
)
parser.add_argument(
    "--switch-after-iterations", type=int, default=np.inf,
    help="How many iterations until learning rate is multiplied by 0.1 or sqrt(e)"
)
parser.add_argument(
    "--loc-lr-multiplier", type=float, default=1,
    help="How much less the localization lr is than the base lr"
)
parser.add_argument(
    "--divide-lr-by", type=float, default=10,
    help="Divide the lr by this number when changing lr"
)
parser.add_argument(
    "--momentum", type=float, default="0",
    help="How large the momentum is. (Use optimizer 'nesterov' for nesterov momentum)"
)
parser.add_argument(
    "--weight-decay", "-w", type=float, default=0,
)
parser.add_argument(
    "--batch-size", '-b', type=int, default=256,
)

epoch_parser = parser.add_mutually_exclusive_group(required=True)
epoch_parser.add_argument(
    "--epochs", '-e', type=int, dest="epochs",
    help="Epochs to train on, default 640" # 640 = 150000 * 256 / 60000
)
epoch_parser.add_argument(
    "--iterations", '-i', type=int, dest="iterations",
    help="Epochs to train on, default 640" # 640 = 150000 * 256 / 60000
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
)
parser.set_defaults(rotate=False)

normalize_parser = parser.add_mutually_exclusive_group(required=False)
normalize_parser.add_argument(
    "--normalize", dest="normalize", action="store_true",
    help="Normalize the data to mean 0 std 1."
)
normalize_parser.add_argument(
    '--no-normalize', dest='normalize', action='store_false',
)
parser.set_defaults(normalize=False)

batchnorm_parser = parser.add_mutually_exclusive_group(required=False)
batchnorm_parser.add_argument(
    "--batchnorm", dest="batchnorm", action="store_true",
    help="Use batchnorm after all STNs."
)
batchnorm_parser.add_argument(
    '--no-batchnorm', dest='batchnorm', action='store_false',
)
parser.set_defaults(batchnorm=False)

args = parser.parse_args()

print("Parsed: ", args)

#%% Read arguments
if args.dataset in data.data_dict:
    print('Using dataset:', args.dataset, '; rotated' if args.rotate else '')
    train_loader, test_loader = data.data_dict[args.dataset](
        rotate = args.rotate,
        batch_size = args.batch_size,
        normalize = args.normalize,
    )
else:
    print('Using precomputed dataset',args.dataset)
    assert args.rotate is False
    train_loader, test_loader = data.get_precomputed(
        path = args.dataset,
        batch_size = args.batch_size,
        normalize = args.normalize,
    )
input_shape = train_loader.dataset[0][0].shape

if args.iterations:
    epochs = args.iterations // len(train_loader)
    print('Rounding',args.iterations,'iterations to',epochs,
          'epochs ==',epochs*len(train_loader),'iterations')
else:
    epochs = args.epochs
    print('Using',epochs,'epochs ==',epochs*len(train_loader),'iterations')
assert epochs > 0

print('Using model:', args.model)
model_class = models.model_dict.get(args.model)
assert not (model_class is None), 'Could not find model'

print('Using localization:', args.localization)
localization_class = models.localization_dict.get(args.localization)
assert localization_class is not None, 'Could not find localization'
assert len(args.stn_placement) > 0 or not localization_class

print('Using optimizer',args.optimizer)
assert args.momentum == 0 or not args.optimizer == 'adam', "Adam can't use momentum."
optimizer_class = {
    'sgd': partial(t.optim.SGD, momentum=args.momentum),
    'nesterov': partial(t.optim.SGD, momentum=args.momentum, nesterov=True),
    'adam': t.optim.Adam
}.get(args.optimizer)
assert not optimizer_class is None, 'Could not find optimizer'

# user must either specify a localization, or keep the
# defaultvalue/assign 0-value to loop and stn_placement
assert localization_class or (not args.loop and (
        not args.stn_placement or args.stn_placement == [0]))

directory = args.name + '/'
makedirs(directory, exist_ok=True)

# Save model details
d = {
        'dataset': args.dataset,
        'model': args.model,
        'model_parameters': args.model_parameters,
        'localization': args.localization,
        'localization_parameters': args.localization_parameters,
        'stn_placement': args.stn_placement,
        'optimizer': args.optimizer,
        'learning_rate': args.lr,
        'switch_after_iterations': args.switch_after_iterations,
        'loc_lr_multiplier': args.loc_lr_multiplier,
        'divide_lr_by': args.divide_lr_by,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'epochs': epochs,
        'loop': args.loop,
        'rotate': args.rotate,
        'normalize': args.normalize,
        'batchnorm': args.batchnorm,
    }
t.save(d, directory + 'model_details')
eval.d = d      # used when calling functions from eval


#%% Setup
print('Will switch learning rate after',args.switch_after_iterations,'iterations',
      'approximately', args.switch_after_iterations / len(train_loader), 'epochs')

def get_scheduler(optimizer):
    return StepLRBase(
        optimizer,
        step_size = args.switch_after_iterations,
        floor_lr = 0, # 0.00005,
        gamma = 1/args.divide_lr_by,
        last_epoch = -1
    )

device = t.device("cuda" if t.cuda.is_available() else "cpu")

final_accuracies = {'train':[], 'test':[]}
# These need to be defined before train and test
optimizer = None
scheduler = None
history = None

is_svhn = 'svhn' in args.dataset.split('/')

cross_entropy = t.nn.CrossEntropyLoss(reduction='mean')
def train(epoch):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        output = model(x)

        if is_svhn:
            loss = sum([cross_entropy(output[i],y[:,i]) for i in range(5)])
            pred = t.stack(output, 2).argmax(1)
            history['train_acc'][epoch] += pred.eq(y).all(1).sum().item()
        else:
            loss = cross_entropy(output, y)
            pred = output.argmax(1, keepdim=True)
            history['train_acc'][epoch] += pred.eq(y.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
        history['train_loss'][epoch] += loss.item() * x.shape[0]

        if batch_idx % 50 == 0 and device == t.device("cpu"):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        scheduler.step()
    history['train_loss'][epoch] /= len(train_loader.dataset)
    history['train_acc'][epoch] /= len(train_loader.dataset)

cross_entropy_sum = t.nn.CrossEntropyLoss(reduction='sum')
def test(epoch = None):
    with t.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            output = model(x)

            if is_svhn:
                loss = sum([cross_entropy_sum(output[i],y[:,i]) for i in range(5)])
                pred = t.stack(output, 2).argmax(1)
                correct += pred.eq(y).all(1).sum().item()
            else:
                loss = cross_entropy_sum(output, y)
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
            test_loss += loss.item()
    
    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)

    if not epoch is None:
        history['test_loss'][epoch] = test_loss
        history['test_acc'][epoch] = correct
    return test_loss, correct

#%% Run
for run in range(args.runs):

    prefix = str(run)

    history = {
        'train_loss': np.zeros(epochs,),
        'test_loss': np.zeros(epochs,),
        'train_acc': np.zeros(epochs,),
        'test_acc': np.zeros(epochs,)
    }

    # Create model
    model = model_class(
        args.model_parameters, input_shape, localization_class,
        args.localization_parameters, args.stn_placement,
        args.loop, args.dataset, args.batchnorm,
    )
    model = model.to(device)

    # Train model
    params = [{'params': model.pre_stn.parameters()},
              {'params': model.final_layers.parameters()},
              {'params': model.output.parameters()}]
    if localization_class:
        params.append({'params': model.localization.parameters(),
                       'lr': args.lr * args.loc_lr_multiplier})

    optimizer = optimizer_class(
        params = params,
        lr = args.lr,
        weight_decay = args.weight_decay,
    )
    scheduler = get_scheduler(optimizer)
    start_time = time.time()

    for epoch in range(epochs):
        if epoch % 100 == 0 and epoch != 0:
            # TODO: ADD SAVING OF OPTIMIZER AND OTHER POTENTIALLY RELEVANT THINGS
            t.save(model.state_dict(), directory + prefix + 'ckpt' + str(epoch))
            print('Saved model')
        train(epoch)
        test(epoch)
        if epoch % 10 == 0:
            print(
                'Epoch', epoch, '\n'
                'Train loss {} acc {} \n Test loss {} acc {}'.format(
                    history['train_loss'][epoch], history['train_acc'][epoch],
                    history['test_loss'][epoch], history['test_acc'][epoch],
            ))
       if epoch % 50 == 0 and localization_class and args.dataset in ['mnist', 'translate', 'scale']:
            res = eval.transformation_statistics(
                model, plot=False, normalize=args.normalize,
                transform = 'rotate' if args.rotate else args.dataset)
            scale = sum(sum(res[-3][label]) for label in range(10)) / len(test_loader.dataset)
            print('x-scaling', scale[0], 'y-scaling', scale[1])

    total_time = time.time() - start_time
    print('Time', total_time)
    print('Time per epoch', total_time / epochs)
    print()

    if localization_class and args.dataset in ['mnist', 'translate', 'scale']:
        res = eval.transformation_statistics(
            model, plot=False, normalize=args.normalize, epochs=10,
            transform = 'rotate' if args.rotate else args.dataset)
        scale = sum(sum(res[-3][label]) for label in range(10)) / (10*len(test_loader.dataset))
        print('x-scaling', scale[0], 'y-scaling', scale[1])
        print()

    if args.dataset in ['mnist', 'translate', 'scale']:
        final_test_accuracy = sum(test()[1] for _ in range(10)) / 10
    else:
        final_test_accuracy = history['test_acc'][-1]

    print('Train accuracy:', history['train_acc'][-1])
    print('Test accuracy:', final_test_accuracy)
    print()
    final_accuracies['train'].append(history['train_acc'][-1])
    final_accuracies['test'].append(final_test_accuracy)

<<<<<<< HEAD
    if localization_class and args.dataset in ['translate', 'mnist']:
        if args.dataset == 'translate':
            res = eval.translation_statistics(
                model, plot=False, all_transformations=True)
        elif args.rotate:
            res = eval.rotation_statistics(
                model, plot=False, all_transformations=True, normalize=args.normalize)
        sx = sum(sum(res[-2][label]) for label in range(10)) / len(test_loader.dataset)
        sy = sum(sum(res[-3][label]) for label in range(10)) / len(test_loader.dataset)
        print('x-scaling', sx, 'y-scaling', sy)

=======
>>>>>>> ad2e1009fa90091c33c35e8c2617118ade00edcd
    t.save(model.state_dict(), directory + prefix + 'final')
    t.save(history, directory + prefix + 'history')

for run in range(args.runs):
    print('Train accuracy:', final_accuracies['train'][run])
    print('Test accuracy:', final_accuracies['test'][run])
