#%%
import numpy as np
import torch as t
import torchvision as tv
import pandas as pd
from skimage import io
import os
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
        transforms = [tv.transforms.ToTensor()]
        if rotate:
            transforms.insert(0,tv.transforms.RandomRotation(90, resample=PIL.Image.BILINEAR))
        if normalize:
            transforms.append(tv.transforms.Normalize((0.1307,), (0.3081,)))
            # subtracts first number and divides with second
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


class CustomDataset(t.utils.data.Dataset):
    """Represents any precomputed dataset."""

    def __init__(self, csv_file, root_dir, transform=None, normalize=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be
                applied on a sample.
            normalize: Whether to normalize the data with the
                mean and std in csv_file
        """ 

        self.frame = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir

        transforms = [tv.transforms.ToTensor()]
        if normalize:
            transforms.append(tv.transforms.Normalize(
                (float(self.frame.iloc[0,0]), self.frame.iloc[0,1], self.frame.iloc[0,2],),
                (float(self.frame.iloc[0,3]), self.frame.iloc[0,4], self.frame.iloc[0,5],),
            ))
        else:
            raise Warning('Not using normalization may lead to values in 0-255')
        if transform:
            transforms.insert(0, transform)
        self.transform = tv.transforms.Compose(transforms)


    def __len__(self):
        return len(self.frame) - 1 # first line contains metadata

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx+1, 0])

        image = io.imread(img_name).astype(np.float32)
        label = np.array(self.frame.iloc[idx+1, 1:]).astype(int)

        image = self.transform(image)

        return (image, label)

def get_precomputed(path, normalize=True):
    directory, csv_file = os.path.split(path)
    images = os.path.join(directory, 'images')
    train_loader = t.utils.data.DataLoader(
        CustomDataset(
            os.path.join(directory, csv_file+'train.csv'),
            images,
            normalize=normalize,
        ),
        batch_size=128, shuffle=True, num_workers=4
    )
    test_loader = t.utils.data.DataLoader(
        CustomDataset(
            os.path.join(directory, csv_file+'test.csv'),
            images,
            normalize=normalize,
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