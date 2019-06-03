#%%
import numpy as np
import torch as t
import torchvision as tv
import PIL

# def oldmnist():
#     (xtrn, ytrn), (xtst, ytst) = k.datasets.mnist.load_data()
#     xtrn = xtrn.reshape([xtrn.shape[0],28,28,1]) / 255
#     ytrn = np.array([[float(y == i) for i in range(10)] for y in ytrn])
#     return (xtrn[:50000],ytrn[:50000],xtrn[50000:],ytrn[50000:])

def mnist(rotate=True,normalize=False):
    transforms = [
        tv.transforms.ToTensor(),
    ]
    if rotate:
        transforms.insert(0,tv.transforms.RandomRotation(90, resample=PIL.Image.BILINEAR))
    if normalize:
        transforms.append(tv.transforms.Normalize((0.1307,), (0.3081,))) # subtracts first number and divides with second
    train_loader = t.utils.data.DataLoader(
        tv.datasets.MNIST(
            root='DATA', train=True, download=True,
            transform=tv.transforms.Compose(transforms)
        ),
        batch_size=256, shuffle=True, num_workers=4
    )
    test_loader = t.utils.data.DataLoader(
        tv.datasets.MNIST(
            root='DATA', train=False, # download absent
            transform=tv.transforms.Compose(transforms)
        ),
        batch_size=256, shuffle=True, num_workers=4
    )
    return (train_loader, test_loader)


def cifar10(rotate=False,normalize=False):
    transforms = [
        tv.transforms.ToTensor(),
    ]
    if rotate:
        transforms.insert(0,tv.transforms.RandomRotation(90, resample=PIL.Image.BILINEAR))
    if normalize:
        transforms.append(tv.transforms.Normalize((0.1307,), (0.3081,))) # WRONG VALUES
    train_loader = t.utils.data.DataLoader(
        tv.datasets.CIFAR10(
            root='DATA', train=True, download=True,
            transform=tv.transforms.Compose(transforms)
        ),
        batch_size=256, shuffle=True, num_workers=4
    )
    test_loader = t.utils.data.DataLoader(
        tv.datasets.CIFAR10(
            root='DATA', train=False, # download absent
            transform=tv.transforms.Compose(transforms)
        ),
        batch_size=256, shuffle=True, num_workers=4
    )
    return (train_loader, test_loader)


data_dic = {
    'mnist':    mnist,
    # 'oldmnist': oldmnist,
    'cifar10':  cifar10
}

cifar10()

#%%
