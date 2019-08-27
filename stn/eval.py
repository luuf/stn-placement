#%%
import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import models
import data
from scipy.misc import imrotate

# directory = "../experiments/mnist/statistics/translate/STFCN0/"
directory = "../experiments/mnist/statistics/STCNN0/"

d = t.load(directory+"model_details")
if d['dataset'] in data.data_dict:
    train_loader, test_loader = data.data_dict[d['dataset']](
        d['rotate'])
else:
    train_loader, test_loader = data.get_precomputed(
        '../'+d['dataset'], normalize=False)

#%% Functions
def get_model(prefix):
    model = models.model_dict[d['model']](
        d['model_parameters'],
        train_loader.dataset[0][0].shape,
        models.localization_dict[d['localization']],
        d['localization_parameters'],
        d['stn_placement'],
        d['loop'],
        d['dataset'],
    )
    model.load_state_dict(t.load(
        directory+str(prefix)+"ckpt100",
        map_location='cpu',
    ))
    # model.load_state_dict(t.load(directory+prefix+"ckpt"+"100"))
    return model

def print_history(prefixes=[0,1,2],loss=False,start=0):
    if type(prefixes) == int:
        prefixes = [prefixes]
    for prefix in prefixes:
        history = t.load(directory + str(prefix) + "history")
        print('PREFIX', prefix)
        print('Max test_acc', np.argmax(history['test_acc']), np.max(history['test_acc']))
        print('Max train_acc', np.argmax(history['train_acc']), np.max(history['train_acc']))
        print('Final test_acc', history['test_acc'][-1])
        print('Final train_acc', history['train_acc'][-1])
        if loss:
            plt.plot(history['train_loss'][start:])
            plt.plot(history['test_loss'][start:])
        else:
            plt.plot(history['train_acc'][start:])
            plt.plot(history['test_acc'][start:])
        plt.show()

def test_stn(model=0, n=4):
    if type(model) == int:
        model = get_model(model)
    model.eval()
    batch = next(iter(test_loader))[0][:n]
    theta = model.localization[0](model.pre_stn[0](batch))
    transformed_batch = model.stn(theta, batch)
    for image,transformed in zip(batch, transformed_batch):
        transformed = transformed.detach()
        if image.shape[0] == 1:
            image = image[0,:,:]
            transformed = transformed[0,:,:]
        else:
            image = np.moveaxis(np.array(image),0,-1)
            transformed = np.moveaxis(transformed.numpy(),0,-1)
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.subplot(1,2,2)
        plt.imshow(transformed)
        plt.show()

def angle_from_matrix(thetas):
    # # V1: Inverts in order to get parameters for the number's
    # #     transformation, and decomposes into Scale Shear Rot
    # mat = F.pad(thetas, (0, 3)).view(-1,3,3)
    # mat[:,2,2] = 1
    # transform = t.inverse(mat)
    # return (np.arctan(transform[:,0,1] / transform[:,0,0])) * 180 / np.pi
    # # negated because the y-axis is inverted, and because I use
    # # counter-clockwise as positive direction

    # V2: Decomposes the window's transformation into Scale Shear Rot.
    #     This Rot*-1 is equal to the inverse's decomposed into Rot Shear Scale.
    thetas = thetas.view(-1,2,3)
    return -(np.arctan(thetas[:,0,1] / thetas[:,0,0])) * 180 / np.pi
    # negated because the images is transformed in the reverse
    # of the predicted transform, because the y-axis is inverted,
    # and because I use counter-clockwise as positive direction

def rotation_statistics(model=0, plot=True):
    if type(model) == int:
        model = get_model(model)

    assert d['rotate']
    _, unrotated_test = data.data_dict[d['dataset']](
        rotate=False, normalize=False) 

    rotated_angles = np.array([])
    predicted_angles = np.array([])

    with t.no_grad():
        model.eval()

        for x, y in unrotated_test:
            angles = np.random.uniform(-90, 90, x.shape[0])
            rot_x = t.tensor([ # pylint: disable=not-callable
                imrotate(im[0], angle) for im, angle in zip(x, angles)
            ], dtype=t.float).reshape(-1, 1, 28, 28)
            # for unfathomable reasons, imrotate converts the image to 0-255
            rot_x = (rot_x - 0.1307 * 255) / (0.3081 * 255) # normalization
            theta = model.localization[0](model.pre_stn[0](rot_x))

            rotated_angles = np.append(rotated_angles, angles)
            predicted_angles = np.append(predicted_angles, angle_from_matrix(theta))

            # # DEBUGGING
            # plt.imshow(x[0,0])
            # plt.colorbar()
            # plt.figure()
            # plt.imshow(rot_x[0,0])
            # plt.colorbar()
            # plt.figure()
            # plt.imshow(model.stn(theta, rot_x)[0,0])
            # plt.colorbar()
            # plt.show()
            # print('rotated', angles[0])
            # print('predicted', angle_from_matrix(theta)[0], flush=True)
            # print('theta', theta[0])
            # raise SystemExit()

    if plot:
        heatmap, xedges, yedges = np.histogram2d(rotated_angles, predicted_angles, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.show()
        
    return rotated_angles, predicted_angles

def distance_from_matrix(thetas):
    thetas = thetas.view((-1,2,3))
    # distances = np.array([
    #     np.linalg.solve(theta[:,0:2], theta[:,2]) for theta in thetas
    # ])
    return np.array(thetas[:,:,2]) * np.array([-1, 1]) * 30
    # both are negated because the digits are transformed in the
    # reverse of predicted transform, and because y is the wrong way

def zooming_statistics(model=0, plot=True):
    if type(model) == int:
        model = get_model(model)

    _, untransformed_test = data.mnist(rotate=False, normalize=False, translate=False)

    translated_distance = np.zeros((0,2))
    predicted_distance = np.zeros((0,2))
    noise = data.MNIST_noise()

    with t.no_grad():
        model.eval()

        for x, y in untransformed_test:
            distance = np.random.randint(-16, 17, (x.shape[0], 2))
            translated = t.zeros(x.shape[0], 1, 60, 60, dtype=t.float)
            for i,(im, d) in enumerate(zip(x, distance)):
                translated[i, 0, 16-d[1] : 44-d[1], 16+d[0] : 44+d[0]] = im[0]
                translated[i] = noise(translated[i])
            
            theta = model.localization[0](model.pre_stn[0](translated))

            translated_distance = np.append(translated_distance, distance, axis=0)
            predicted_distance = np.append(predicted_distance, distance_from_matrix(theta), axis=0)

            # # DEBUGGING
            # plt.imshow(x[0,0])
            # plt.figure()
            # plt.imshow(translated[0,0])
            # plt.figure()
            # plt.imshow(model.stn(theta, translated)[0,0])
            # plt.show()
            # print('translated', distance[0])
            # print('predicted', distance_from_matrix(theta)[0], flush=True)
            # raise SystemExit()

    if plot:
        heatmap, xedges, yedges = np.histogram2d(translated_distance[:,0], predicted_distance[:,0], bins=(33,33))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.figure()

        heatmap, xedges, yedges = np.histogram2d(translated_distance[:,1], predicted_distance[:,1], bins=(33,33))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.show()

    return translated_distance, predicted_distance

#%%
