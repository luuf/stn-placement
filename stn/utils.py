import numpy as np
import skimage
import torch

def get_moment_angle(ims):
    ims = ims.numpy()
    thetas = []
    for image in ims.reshape([ims.shape[0], *ims.shape[-2:]]):
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
        thetas.append(theta)
    return np.array(thetas) if len(thetas) > 1 else thetas[0]

def angle_from_matrix(thetas):
    # V2: Decomposes the window's transformation into Scale Shear Rot.
    #     This Rot*-1 is equal to the inverse's decomposed into Rot Shear Scale.
    thetas = thetas.view(-1,2,3)
    return -(torch.atan(thetas[:,0,1] / thetas[:,0,0])) # * 180 / np.pi
    # negated because the images is transformed in the reverse
    # of the predicted transform, because the y-axis is inverted,
    # and because I use counter-clockwise as positive direction

def matrix_from_angle(thetas):
    z = torch.zeros(len(thetas),1)
    thetas = thetas.reshape(-1,1)
    return torch.cat((torch.cos(thetas), -torch.sin(thetas), z,
                      torch.sin(thetas),  torch.cos(thetas), z,), dim=1)
                      
def matrix_from_moment(ims):
    angles = -torch.tensor(get_moment_angle(ims), dtype=torch.float32)
    return matrix_from_angle(angles).reshape(-1,2,3)
    