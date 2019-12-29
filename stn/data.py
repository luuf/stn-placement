#%%
import numpy as np
import torch as t
import torchvision as tv
import torchvision.transforms.functional as tvF
import pandas as pd
import os
import PIL
import skimage
import matplotlib.pyplot as plt

# def oldmnist():
#     (xtrn, ytrn), (xtst, ytst) = k.datasets.mnist.load_data()
#     xtrn = xtrn.reshape([xtrn.shape[0],28,28,1]) / 255
#     ytrn = np.array([[float(y == i) for i in range(10)] for y in ytrn])
#     return (xtrn[:50000],ytrn[:50000],xtrn[50000:],ytrn[50000:])

#mnist_root = '/local_storage/datasets/'
mnist_root = 'data/cache'

class MNIST_noise:
    """Adds noise to images of size 60x60. The noise is generated by
    taking 6 random 6x6 crops of mnist images and adding them to random
    places in the image.
    """
    def __init__(self, imsize=60, scale=False):
        self.imsize = imsize
        self.scale = scale
        # padding = (imsize-6) // 2
        transform = tv.transforms.Compose([
            tv.transforms.RandomCrop(6),
            # tv.transforms.Pad(padding),
            # tv.transforms.RandomAffine(0,translate=(padding/imsize,padding/imsize)),
            tv.transforms.ToTensor()
        ])
        self.data = t.utils.data.DataLoader(
            tv.datasets.MNIST(
                root=mnist_root, train=True, download=True, transform=transform),
            batch_size=6, shuffle=True, num_workers=0, drop_last=True)
        self.dataiter = iter(self.data)
    
    def __call__(self, img):
        try:
            noise = next(self.dataiter)[0]
        except StopIteration:
            self.dataiter = iter(self.data)
            noise = next(self.dataiter)[0]
        if self.scale:
            imloc = np.random.randint(2,self.imsize-2,(6,2))
            sizes = np.random.randint(1, 10, 6)
            for i,((ix,iy),s) in enumerate(zip(imloc,sizes)):
                s = min(s, self.imsize-max(ix,iy), min(ix,iy))
                loc = img[:, ix-s:ix+s, iy-s:iy+s]
                # print(t.nn.functional.interpolate(noise[i:i+1], size=(1,1,2*s,2*s), mode='bilinear')[0])
                loc += t.nn.functional.interpolate(noise[i:i+1], size=(2*s,2*s), mode='bilinear')[0]
                t.clamp(loc, max=1, out=loc)
                # img[:, ix:ix+6, iy:iy+6] += im
                # t.clamp(img[0, ix:ix+6, iy:iy+6], max=1, out=img[0, ix:ix+6, iy:iy+6])
        else:
            imloc = np.random.randint(0,self.imsize-6,(6,2))
            for im,(ix,iy) in zip(noise,imloc):
                img[:, ix:ix+6, iy:iy+6] += im
                t.clamp(img[0, ix:ix+6, iy:iy+6], max=1, out=img[0, ix:ix+6, iy:iy+6])
            
        return img

class RandomScale:
    def __init__(self, loglow, loghigh):
        self.loglow = loglow
        self.loghigh = loghigh

    def __call__(self, img):
        return tvF.affine(
            img, angle=0, translate=(0,0), shear=0,
            scale = 2**(np.random.uniform(self.loglow, self.loghigh)),
            resample = PIL.Image.BILINEAR,
            fillcolor = 0)

# http://raphael.candelier.fr/?blog=Image%20Moments
def moment_rotate(im):
    image = im.numpy()[0]
    # plt.imshow(image)
    m = skimage.measure.moments(image, 2)
    x, y = m[1,0]/m[0,0], m[0,1]/m[0,0]
    ux = m[2,0]/m[0,0] - x**2
    uy = m[0,2]/m[0,0] - y**2
    um = m[1,1]/m[0,0] - x*y
    theta = np.arctan(2*um/(ux-uy))/2 + (ux<uy)*np.pi/2 # + np.pi/4
    # print('Theta', theta*180/np.pi)

    i,j = np.indices(image.shape).astype(np.float32)
    i *= np.cos(theta)
    j *= np.sin(theta)
    d = np.cos(theta)*x + np.sin(theta)*y
    matrix = ((j+i-d) > 0).astype(np.float32)
    # temp = matrix[int(x)][int(y)]
    # matrix[int(x)][int(y)] = 2
    # plt.figure()
    # plt.imshow(matrix)
    # matrix[int(x)][int(y)] = temp
    matrix -= 0.5
    if np.sum(image*matrix) > 0:
        theta += np.pi
        # print('Added')
    # print('Theta', theta*180/np.pi)
    theta += np.pi/4

    im = tvF.to_pil_image(im)
    im = tvF.rotate(im, -theta*180/np.pi, resample=PIL.Image.BILINEAR, center=(y,x))
    im = tvF.to_tensor(im)
    # plt.figure()
    # plt.imshow(im[0])
    return im


def mnist(rotate=True, normalize=True, translate=False, scale=False, batch_size=256):
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
        scale (bool): If True, the mnist digits are scaled a random amount.
        batch_size (int): The size of minibatches.
    """
    if translate:
        transforms = [
            tv.transforms.Pad(16), # (60-28)/2 = 16
            tv.transforms.RandomAffine(degrees=0, translate=(16/60,16/60)),
            tv.transforms.ToTensor(),
            MNIST_noise(60),
        ]
        if normalize:
            transforms.append(tv.transforms.Normalize((0.0363,), (0.1870,)))
            # approximate numbers derived from the transformation process
            # empirically, this should actually be ((0.0394,), (0.1798))
    elif scale:
        transforms = [
            tv.transforms.Pad(3*28//2),
            RandomScale(-1, 2),
            tv.transforms.ToTensor(),
            MNIST_noise(112, scale=True),
        ]
        if normalize:
            transforms.append(tv.transforms.Normalize((0.0414,), (0.1751,)))
    else:
        random_moment_rotate = tv.transforms.RandomApply([moment_rotate], p=1)
        transforms = [tv.transforms.ToTensor(), random_moment_rotate]
        if rotate:
            transforms.insert(0,tv.transforms.RandomRotation(90, resample=PIL.Image.BILINEAR))
        if normalize:
            transforms.append(tv.transforms.Normalize((0.1307,), (0.3081,)))
            # subtracts first number and divides with second
            # empirically, this should actually be ((0.1302,), (0.2892))
    train_loader = t.utils.data.DataLoader(
        tv.datasets.MNIST(
            root=mnist_root, train=True, download=True,
            transform=tv.transforms.Compose(transforms)
        ),
        batch_size=batch_size, shuffle=True, num_workers=16 if translate or scale else 4
    )
    test_loader = t.utils.data.DataLoader(
        tv.datasets.MNIST(
            root=mnist_root, train=False, # download absent
            transform=tv.transforms.Compose(transforms)
        ),
        batch_size=batch_size, shuffle=True, num_workers=16 if translate or scale else 4
    )
    def set_moment_probability(p):
        random_moment_rotate.p = p
    train_loader.dataset.set_moment_probability = set_moment_probability
    test_loader.dataset.set_moment_probability = set_moment_probability
    # note that this prevents us from modifying only the train/test loader
    return (train_loader, test_loader)

def translated_mnist(rotate=False,normalize=False, batch_size=256):
    "Helper function to call mnist with certain variables"
    assert rotate is False
    return mnist(rotate=False, normalize=normalize, translate=True, batch_size=batch_size)

def scaled_mnist(rotate=False, normalize=False, batch_size=256):
    "Helper function to call mnist with certain variables"
    assert rotate is False
    return mnist(rotate=False, normalize=normalize, translate=False, scale=True, batch_size=batch_size)


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

# inspired by data/plankton/rescale_images.py
class Resize_Image:
    def __init__(self, length, scale_up=False, invert=True):
        self.length = length
        self.scale_up = scale_up
        self.invert = invert

    def __call__(self, im):
        if self.invert:
            im = PIL.ImageOps.invert(im)
        ratio = self.length / max(im.size)

        if ratio < 1 or self.scale_up:
            new_size = np.rint(ratio * np.array(im.size[:2])).astype(int)
            im = im.resize(new_size, PIL.Image.ANTIALIAS)
        else:
            new_size = np.array(im.size)
        dw, dh = self.length - new_size
        padding = (dw//2, dh//2, dw - dw//2, dh - dh//2)
        im = PIL.ImageOps.expand(im, padding)

        return im

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

        self.moment_rotate = tv.transforms.RandomApply([moment_rotate], p=1)
        transforms = [tv.transforms.ToTensor(), self.moment_rotate]

        if transform:
            transforms.insert(0, transform)

        if normalize:
            if len(self.frame.columns) == 6 and self.frame.iloc[0,3] != 0:
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

    def set_moment_probability(self, p):
        self.moment_rotate.p = p

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
    plankton = 'plankton' in path.split('/')
    if plankton:
        transform = tv.transforms.Compose([
            tv.transforms.RandomAffine(
                degrees = (-180,180),
                translate = (10/64, 10/64),
                shear = (-20,20),
                resample = PIL.Image.BILINEAR),
            RandomScale(np.log2(1/1.6), np.log2(1.6)),
            tv.transforms.RandomHorizontalFlip(0.5),
            # original also does 'stretching'
        ])
    else:
        transform = None
    directory, csv_file = os.path.split(path)
    if csv_file == 'unprocessed_':
        images = os.path.join(directory, 'unprocessed')
        resize192 = Resize_Image(192, scale_up=True)
        resize = Resize_Image(64, scale_up=True, invert=False)
        transform = (resize if transform is None
                    else tv.transforms.Compose([resize192, transform, resize]))
        test_transform = Resize_Image(64, scale_up=True)
    else:
        images = os.path.join(directory, 'images')
        test_transform = None
    train_loader = t.utils.data.DataLoader(
        CustomDataset(
            os.path.join(directory, csv_file+'train.csv'),
            root_dir=images, normalize=normalize, transform=transform
        ),
        batch_size=batch_size, shuffle=True, num_workers= 16 if plankton else 4
    )
    test_loader = t.utils.data.DataLoader(
        CustomDataset(
            os.path.join(directory, csv_file+'test.csv'),
            root_dir=images, normalize=normalize, transform=test_transform
        ),
        batch_size=batch_size, shuffle=True, num_workers=4
    )
    return (train_loader, test_loader)

data_dict = {
    'mnist': mnist,
    'translate': translated_mnist,
    'scale': scaled_mnist,
    'cifar10': cifar10,
    'augment': augmented_cifar,
}

#%%
