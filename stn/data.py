#%%
import numpy as np
import torch as t
import torchvision as tv
import pandas as pd
import os
import PIL
from functools import partial

# def oldmnist():
#     (xtrn, ytrn), (xtst, ytst) = k.datasets.mnist.load_data()
#     xtrn = xtrn.reshape([xtrn.shape[0],28,28,1]) / 255
#     ytrn = np.array([[float(y == i) for i in range(10)] for y in ytrn])
#     return (xtrn[:50000],ytrn[:50000],xtrn[50000:],ytrn[50000:])

class MNIST_noise:
    """Adds noise to images of size 60x60. The noise is generated by
    taking 6 random 6x6 crops of mnist images and adding them to random
    places in the image.
    """
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
        return t.clamp(img, max=1)

def mnist(rotate=True, normalize=True, translate=False, batch_size=256):
    """Gets the mnist dataset from torchvision, and manipulates it.
    Args:
        rotate (bool): Whether to rotate the images a random amount
            between -90 and 90 degrees.
        normalize (bool): Whether to normalize the values to mean 0,
            std 1. If False, the values are between 0 and 1.
        translate (bool): If True, the mnist digits are placed in a
            random location of a black 60x60 image, and MNIST_noise
            is applied to each image. If translate is True, no rotation
            or normalization is performed.
    """
    if translate:
        transforms = [
            tv.transforms.Pad(16), # (60-28)/2 = 16
            tv.transforms.RandomAffine(degrees=0, translate=(16/60,16/60)),
            tv.transforms.ToTensor(),
            MNIST_noise(),
        ]
        if normalize:
            transforms.append(tv.transforms.Normalize((0.0363,), (0.1870,)))
            # approximate numbers derived from the transformation process
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
        batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = t.utils.data.DataLoader(
        tv.datasets.MNIST(
            root='data/cache', train=False, # download absent
            transform=tv.transforms.Compose(transforms)
        ),
        batch_size=batch_size, shuffle=True, num_workers=4
    )
    return (train_loader, test_loader)

def translated_mnist(rotate=False,normalize=False, batch_size=256):
    "Helper function to call mnist with certain variables"
    assert rotate is False
    return mnist(False, normalize, True, batch_size=batch_size)


def cifar10(rotate=False,normalize=False,augment=False, batch_size=256):
    """Gets the cifar10 dataset from torchvision, and manipulates it.
    Args:
        rotate (bool): Whether to rotate the images a random amount
            between -90 and 90 degrees.
        normalize (bool): Should correspond to normalizing all the
            values. However, this isn't implemented, so it should
            always be false.
        augment (bool): If True, the images are randomly flipped,
            rotated, translated, scaled, and sheared each time they're
            used.
    """
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
        batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = t.utils.data.DataLoader(
        tv.datasets.CIFAR10(
            root='data/cache', train=False, # download absent
            transform=tv.transforms.Compose(test_transforms)
        ),
        batch_size=batch_size, shuffle=True, num_workers=4
    )
    return (train_loader, test_loader)

def augmented_cifar(rotate=False,normalize=False):
    "Helper function to call cifar with certain variables"
    assert normalize is False
    return cifar10(rotate,False,True)


class CustomDataset(t.utils.data.Dataset):
    """Represents any precomputed dataset. The first row of the csv
    file should contain the mean and the standard deviations of all
    channels in the images, so that the dataset can normalize them.
    After that, the first column denotes the path to the images
    relative root_dir, and the rest of the columns denotes the labels.

    Args:
        csv_file (string): The path to a csv file.
        root_dir (string): The path to a directory containing images.
        transform (callable or None): Optional transform to be
            applied on a sample.
        normalize (bool): Whether to normalize the data with the
            mean and std in csv_file
    """ 

    def __init__(self, csv_file, root_dir, transform=None, normalize=True):

        self.frame = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir

        if transform:
            self.transform = transform

        else:
            transforms = [tv.transforms.ToTensor()]

            if normalize:
                if len(self.frame.columns) == 6:
                    transforms.append(tv.transforms.Normalize(
                        (float(self.frame.iloc[0,0])/255, self.frame.iloc[0,1]/255, self.frame.iloc[0,2]/255,),
                        (float(self.frame.iloc[0,3])/255, self.frame.iloc[0,4]/255, self.frame.iloc[0,5]/255,),
                    ))
                else:
                    transforms.append(tv.transforms.Normalize(
                        (float(self.frame.iloc[0,0])/255,),
                        (float(self.frame.iloc[0,1])/255,),
                    ))

            self.transform = tv.transforms.Compose(transforms)


    def __len__(self):
        return len(self.frame) - 1 # first line contains metadata

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx+1, 0])

        image = PIL.Image.open(img_name)
        label = np.array(self.frame.iloc[idx+1, 1:]).astype(int)
        if len(label) == 1:
            label = label[0]

        image = self.transform(image)

        return (image, label)

def get_precomputed(path, normalize=True, batch_size=128):
    """Creates a custom dataset from images saved to file and csv
    files describing the images. The images must be saved in a folder
    named `images` in the same folder as the csv files.

    Args:
        path (string): A prefix to two csv files. This path
            concatenated with train.csv should be the path to a csv
            file containing information about the training data;
            concatenated with test.csv should be the path to a csv
            file containing information about the test data.
    """
    gray = True
    if gray and normalize:
        transform = tv.transforms.Compose([
            # partial(tv.transforms.functional.adjust_gamma, gamma=1/2.2),
            tv.transforms.Grayscale(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((.4352,), (.1981,)) # for luminosity
            # partial(t.mean, dim=0, keepdim=True),
            # tv.transforms.Normalize((.4398,), (0.1956,)) # for intensity
        ])
    else:
        transform = None
    directory, csv_file = os.path.split(path)
    images = os.path.join(directory, 'images')
    train_loader = t.utils.data.DataLoader(
        CustomDataset(
            os.path.join(directory, csv_file+'train.csv'),
            root_dir=images, normalize=normalize, transform=transform
        ),
        batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = t.utils.data.DataLoader(
        CustomDataset(
            os.path.join(directory, csv_file+'test.csv'),
            root_dir=images, normalize=normalize, transform=transform
        ),
        batch_size=batch_size, shuffle=True, num_workers=4
    )
    return (train_loader, test_loader)

data_dict = {
    'mnist': mnist,
    'translate': translated_mnist,
    'cifar10': cifar10,
    'augment': augmented_cifar,
}

#%%