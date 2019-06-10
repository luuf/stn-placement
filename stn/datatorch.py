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

def mnist(rotate=True,normalize=True,translate=False):
    if translate:
        transforms = [
            tv.transforms.Pad(16), # (60-28)/2 = 16
            tv.transforms.RandomAffine(degrees=0, translate=(16/60,16/60)),
            tv.transforms.ToTensor()
        ]
    else:
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

def translated_mnist(rotate=False,normalize=False):
    assert rotate is False
    assert normalize is False
    return mnist(False,False,True)


def cifar10(rotate=False,normalize=False,augment=False):
    if normalize:
        raise Exception("Normalization not implemented")
    if augment:
        train_transforms = [
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomAffine(
                degrees = 5,
                translate = (0.05,0.05),
                scale = (0.9,1.1),
                shear = 5,
                resample = PIL.Image.BILINEAR,
            ),
            tv.transforms.ToTensor(),
        ]
        test_transforms = [tv.transforms.ToTensor()]
    else:
        train_transforms = [tv.transforms.ToTensor()]
        test_transforms = [tv.transforms.ToTensor()]
    if rotate:
        train_transforms.insert(0,tv.transforms.RandomRotation(90, resample=PIL.Image.BILINEAR))
        test_transforms.insert(0,tv.transforms.RandomRotation(90, resample=PIL.Image.BILINEAR))
    train_loader = t.utils.data.DataLoader(
        tv.datasets.CIFAR10(
            root='DATA', train=True, download=True,
            transform=tv.transforms.Compose(train_transforms)
        ),
        batch_size=256, shuffle=True, num_workers=4
    )
    test_loader = t.utils.data.DataLoader(
        tv.datasets.CIFAR10(
            root='DATA', train=False, # download absent
            transform=tv.transforms.Compose(test_transforms)
        ),
        batch_size=256, shuffle=True, num_workers=4
    )
    return (train_loader, test_loader)

def augmented_cifar(rotate=False,normalize=False,augment=False):
    assert normalize is False
    return cifar10(rotate,False,True)

data_dic = {
    'mnist': mnist,
    'translate': translated_mnist,
    'cifar10': cifar10,
    'augment': augmented_cifar,
}

#%%
