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

class MNIST_noise:
    def __init__(self):
        self.data = tv.datasets.MNIST(
            root='data/cache', train=True, download=True,
        )
        self.transform = tv.transforms.Compose([
            tv.transforms.RandomCrop(6),
            tv.transforms.Pad(27),
            tv.transforms.RandomAffine(0,translate=(27/60,27/60)),
            tv.transforms.ToTensor()
        ])
    
    def __call__(self, img):
        indices = np.random.randint(0,60000,6)
        for i in indices:
            img += self.transform(self.data[i][0])
        return t.clamp(img, max=1) # pylint: disable=no-member

def mnist(rotate=True,normalize=True,translate=False):
    if translate:
        transforms = [
            tv.transforms.Pad(16), # (60-28)/2 = 16
            tv.transforms.RandomAffine(degrees=0, translate=(16/60,16/60)),
            tv.transforms.ToTensor(),
            MNIST_noise(),
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
            root='data/cache', train=True, download=True,
            transform=tv.transforms.Compose(transforms)
        ),
        batch_size=256, shuffle=True, num_workers=4
    )
    test_loader = t.utils.data.DataLoader(
        tv.datasets.MNIST(
            root='data/cache', train=False, # download absent
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
            root='data/cache', train=True, download=True,
            transform=tv.transforms.Compose(train_transforms)
        ),
        batch_size=256, shuffle=True, num_workers=4
    )
    test_loader = t.utils.data.DataLoader(
        tv.datasets.CIFAR10(
            root='data/cache', train=False, # download absent
            transform=tv.transforms.Compose(test_transforms)
        ),
        batch_size=256, shuffle=True, num_workers=4
    )
    return (train_loader, test_loader)

def augmented_cifar(rotate=False,normalize=False):
    assert normalize is False
    return cifar10(rotate,False,True)

def get_precomputed(name, normalize=True):
    try:
        d = np.load('data/'+name+'.npz')
    except FileNotFoundError:
        d = np.load('../data/'+name+'.npz')
    if normalize:
        m = np.mean(d['trn_x'], (0,2,3)).reshape(1,3,1,1)
        s = np.std(d['trn_x'], (0,2,3)).reshape(1,3,1,1)
        trn_x = (d['trn_x']-m) / s
        tst_x = (d['tst_x']-m) / s
    else:
        trn_x = d['trn_x']
        tst_x = d['tst_x']
    train_loader = t.utils.data.DataLoader(
        t.utils.data.TensorDataset(
            t.tensor(trn_x),
            t.tensor(d['trn_y']),
        ),
        batch_size=128, shuffle=True, num_workers=4
    )
    test_loader = t.utils.data.DataLoader(
        t.utils.data.TensorDataset(
            t.tensor(tst_x),
            t.tensor(d['tst_y']),
        ),
        batch_size=128, shuffle=True, num_workers=4
    )
    return (train_loader, test_loader)

data_dict = {
    'mnist': mnist,
    'translate': translated_mnist,
    'cifar10': cifar10,
    'augment': augmented_cifar,
}

#%%
#%%
