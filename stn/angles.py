import numpy as np
import skimage
import torch

def direction_of_skew(image, x, y, theta):
    i,j = np.indices(image.shape).astype(np.float32)
    i *= np.cos(theta)
    j *= np.sin(theta)
    d = np.cos(theta)*x + np.sin(theta)*y
    matrix = ((j+i-d)**3).astype(np.float32)
    # matrix -= 0.5
    # temp = matrix[int(x)][int(y)]
    # matrix[int(x)][int(y)] = 2
    # plt.figure()
    # plt.imshow(matrix)
    # matrix[int(x)][int(y)] = temp
    return np.sum(image*matrix) > 0


def get_moment_angle(ims, flip=False):
    ims = ims.numpy()
    thetas = []
    flips = []
    for image in ims.reshape([ims.shape[0], *ims.shape[-2:]]):
        # plt.imshow(image)
        m = skimage.measure.moments(image, 2)
        x, y = m[1,0]/m[0,0], m[0,1]/m[0,0]
        ux = m[2,0]/m[0,0] - x**2
        uy = m[0,2]/m[0,0] - y**2
        um = m[1,1]/m[0,0] - x*y
        theta = np.arctan(2*um/(ux-uy))/2 + (ux<uy)*np.pi/2
        # print('Theta', theta*180/np.pi)

        if direction_of_skew(image, x, y, theta):
            theta += np.pi

        if flip:
            if direction_of_skew(image, x, y, theta+np.pi/2):
                flips.append(True)
                theta += np.pi/2
            else:
                flips.append(False)

        theta += np.pi/4
        thetas.append(theta)

    if flip:
        return (np.array(thetas), np.array(flips)) if len(thetas) > 1 else (thetas[0], flip)
    else:
        return np.array(thetas) if len(thetas) > 1 else thetas[0]

def angle_from_matrix(thetas):
    # V2: Decomposes the window's transformation into Scale Shear Rot.
    #     This Rot*-1 is equal to the inverse's decomposed into Rot Shear Scale.
    thetas = thetas.view(-1,2,3)
    return -(torch.atan(thetas[:,0,1] / thetas[:,0,0])) # * 180 / np.pi
    # negated because the images is transformed in the reverse
    # of the predicted transform, because the y-axis is inverted,
    # and because I use counter-clockwise as positive direction

def matrix_from_angle(thetas, flips=None):
    z = torch.zeros(len(thetas),1)
    thetas = thetas.reshape(-1,1)
    signs = (torch.ones(len(thetas),1) if flips is None
        else torch.tensor(-2*(flips-0.5)).reshape(len(thetas),1).float())
    return torch.cat((torch.cos(thetas),signs*-torch.sin(thetas), z,
                      torch.sin(thetas),signs*torch.cos(thetas), z,), dim=1)
                      
def matrix_from_moment(ims):
    angles, flips = get_moment_angle(ims, flip=True)
    angles = -torch.tensor(angles, dtype=torch.float32)
    return matrix_from_angle(angles, flips).reshape(-1,2,3)
    