#%% Import
import torch as t
import torchvision as tv
import numpy as np
import time
import data

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--dataset", '-d', type=str,
    help="Dataset to save"
)
parser.add_argument(
    "--rotate", type=bool, default=False,
    help="Whether to rotate the images before saving"
)
parser.add_argument(
    "--name", '-n', type=str,
    help="Name of directory to save the data in"
)
args = parser.parse_args()

train_loader, test_loader = data.data_dict[args.dataset](args.rotate)

t = time.time()

l = len(train_loader.dataset)
train_imgs = np.zeros((l,)+train_loader.dataset[0][0].shape,dtype=np.float32)
train_targets = np.zeros(l,dtype=np.int)
for i,(img,target) in enumerate(train_loader.dataset):
    train_imgs[i] = img
    train_targets[i] = target
    if i % 100 == 0:
        print('train',i)

l = len(test_loader.dataset)
test_imgs = np.zeros((l,)+test_loader.dataset[0][0].shape,dtype=np.float32)
test_targets = np.zeros(l,dtype=np.int)
for i,(img,target) in enumerate(test_loader.dataset):
    test_imgs[i] = img
    test_targets[i] = target
    if i % 100 == 0:
        print('test',i)

print(time.time() - t)

print('Saving as','DATA/'+args.name)
np.savez_compressed('DATA/'+args.name, 
    trn_x=train_imgs, trn_y=train_targets,
    tst_x=test_imgs,  tst_y=test_targets,
)
