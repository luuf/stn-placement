#%%
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import data

def test_transform(theta, length=17, im=None):
    if im is None:
        im = torch.zeros((1,1,length,length))
        im[0,0,0,0] = 1
        im[0,0,0,length - 1] = 2
        im[0,0,length - 1,0] = 3
        im[0,0,length - 1,length - 1] = 4
        im[0,0,(length - 1)//2, (length - 1)//2] = 5
    grid = F.affine_grid(torch.tensor(theta).view(-1,2,3), im.shape)
    transformed = F.grid_sample(im, grid)
    return transformed

train,test = data.mnist(rotate=False,normalize=False,translate=False)
test_im = train.dataset[0][0].view(1,1,28,28)
plt.imshow(test_im[0,0])
plt.show()

identity = [1.,0,0,0,1,0]
theta = torch.tensor([0.3,-1,-.5,1.3,-0.2,0.8]).view(-1,2,3)
# distance = np.linalg.solve(theta[0,:,0:2], theta[0,:,2])
distance = theta[0,:,2]
print('distance', distance)

trans = test_transform([1., 0, distance[0], 0, 1, distance[1]],im=test_im)
plt.imshow(trans[0,0])
plt.colorbar()
plt.figure()
plt.imshow(test_transform([theta[0,0,0], theta[0,0,1], 0, theta[0,1,0], theta[0,1,1], 0], im=trans)[0,0])
plt.colorbar()
plt.figure()

plt.imshow(test_transform(theta,im=test_im)[0,0])
plt.colorbar()
plt.show()
#%%
