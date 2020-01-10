#%% Import
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import numpy as np
import time
import eval
from os import makedirs, path
import data
import models
import parser
import angles
from datetime import datetime
from functools import partial
from ylvas_code.lrdecay_functions import StepLRBase

print('Launched at', datetime.now())
#%% Parse arguments
args = parser.get_parser().parse_args()

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
    'sgd': partial(optim.SGD, momentum=args.momentum),
    'nesterov': partial(optim.SGD, momentum=args.momentum, nesterov=True),
    'adam': optim.Adam
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
        'iterative': args.iterative,
        'pretrain': args.pretrain,
    }
torch.save(d, directory + 'model_details')
eval.d = d      # used when calling functions from eval


#%% Setup
# switch_after_epochs = [it/len(train_loader) for it in args.switch_after_iterations]
print('Will switch learning rate after',args.switch_after_iterations,'iterations',
      'approximately', [it/len(train_loader) for it in args.switch_after_iterations], 'epochs')

def get_scheduler(optimizer):
    if len(args.switch_after_iterations) == 1:
        return optim.lr_scheduler.StepLR(
            optimizer = optimizer,
            step_size = args.switch_after_iterations[0],
            gamma = 1/args.divide_lr_by
        )
    return optim.lr_scheduler.MultiStepLR(
        optimizer = optimizer,
        milestones = args.switch_after_iterations,
        gamma = 1/args.divide_lr_by,
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

final_accuracies = {'train':[], 'test':[]}
# These need to be defined before train and test
optimizer = None
scheduler = None
history = None

is_svhn = 'svhn' in args.dataset.split('/')

cross_entropy = nn.CrossEntropyLoss(reduction='mean')
def train(epoch):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        output = model(x)

        if is_svhn:
            loss = sum([cross_entropy(output[i],y[:,i]) for i in range(5)])
            pred = torch.stack(output, 2).argmax(1)
            history['train_acc'][epoch] += pred.eq(y).all(1).sum().item()
        else:
            loss = cross_entropy(output, y)
            pred = output.argmax(1, keepdim=True)
            history['train_acc'][epoch] += pred.eq(y.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
        history['train_loss'][epoch] += loss.item() * x.shape[0]

        if batch_idx % 50 == 0 and device == torch.device("cpu"):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        scheduler.step()
    history['train_loss'][epoch] /= len(train_loader.dataset)
    history['train_acc'][epoch] /= len(train_loader.dataset)

cross_entropy_sum = nn.CrossEntropyLoss(reduction='sum')
def test(epoch = None):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            output = model(x)

            if is_svhn:
                loss = sum([cross_entropy_sum(output[i],y[:,i]) for i in range(5)])
                pred = torch.stack(output, 2).argmax(1)
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

def pretrain(epoch):
    model.train()
    for batch_idx, (x, _) in enumerate(train_loader):
        moment_theta = angles.matrix_from_moment(x)[:,0:2,0:2].to(device)
        x = x.to(device)
        optimizer.zero_grad()

        theta = model(x)[:,0:2,0:2]

        #stn_angle = angles.angle_from_matrix(theta)
        #loss = t.mean(((stn_angle-moment_angle)%(2*np.pi))**2)
        det = torch.abs(torch.det(theta))
        loss = torch.mean(torch.abs(theta-moment_theta)) + torch.mean(det + 1/det) - 2
        loss.backward()
        optimizer.step()
        history['train_loss'][epoch] += loss.item() * x.shape[0]

        if batch_idx % 50 == 0 and device == torch.device("cpu"):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        scheduler.step()
    history['train_loss'][epoch] /= len(train_loader.dataset)

def pretest(epoch):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for x, _ in test_loader:
            moment_theta = angles.matrix_from_moment(x)[:,0:2,0:2].to(device)
            x = x.to(device)

            theta = model(x)[:,0:2,0:2]

            det = torch.abs(torch.det(theta))
            loss = torch.sum(torch.abs(theta-moment_theta)) + 4*torch.sum(det + 1/det) - 8
            test_loss += loss.item()
    
    test_loss /= 4*len(test_loader.dataset)

    if not epoch is None:
        history['test_loss'][epoch] = test_loss
    return test_loss

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
        args.loop, args.dataset, args.batchnorm, args.iterative
    )
    model = model.to(device)
    model.pretrain = args.pretrain

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

    train_loader.dataset.set_moment_probability(0)
    test_loader.dataset.set_moment_probability(0)
    for epoch in range(epochs):
        if epoch % 100 == 0 and epoch != 0:
            # TODO: ADD SAVING OF OPTIMIZER AND OTHER POTENTIALLY RELEVANT THINGS
            torch.save(model.state_dict(), directory + prefix + 'ckpt' + str(epoch))
        if args.pretrain:
            pretrain(epoch)
            pretest(epoch)
        else:
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

    torch.save(model.state_dict(), directory + prefix + 'final')
    torch.save(history, directory + prefix + 'history')

for run in range(args.runs):
    print('Train accuracy:', final_accuracies['train'][run])
    print('Test accuracy:', final_accuracies['test'][run])
