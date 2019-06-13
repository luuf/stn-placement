import numpy as np
import torch as t

def rotate_tensor(im,rad=None):
    assert len(im.shape) == 4
    if rad is None:
        rad = np.random.uniform(
            low = -np.pi/2,
            high = np.pi/2,
            size = im.shape[0]
        )
    c = np.cos(rad)
    s = np.sin(rad)
    theta = [c, -s, 0, s, c, 0]
    grid = t.nn.functional.affine_grid(theta, im.size())
    return t.nn.functional.grid_sample(im, grid)