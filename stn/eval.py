#%%
import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import models
import data
from skimage.transform import rotate
import copy
from os import path
from functools import partial
from importlib import reload

directory, d, train_loader, test_loader, untransformed_test = None, None, None, None, None

#%% Functions
def load_data(data_dir, normalize=True):
    global directory, d, train_loader, test_loader, untransformed_test
    directory = '../experiments/'+data_dir

    d = t.load(directory+"/model_details")
    if 'normalize' in d:
        normalize = d['normalize']
    elif d['dataset'] == 'translate':
        print('Assuming no normalizaion on translated data')
        normalize = False

    if d['dataset'] in data.data_dict:
        train_loader, test_loader = data.data_dict[d['dataset']](
            rotate = d['rotate'], normalize = normalize)
        if d['rotate']:
            _, untransformed_test = data.data_dict[d['dataset']](
                rotate=False, normalize=False)
        elif d['dataset'] == 'translate':
            _, untransformed_test = data.mnist(rotate=False, normalize=False)
    else:
        try:
            train_loader, test_loader = data.get_precomputed(
                '../'+d['dataset'], normalize=normalize)
        except FileNotFoundError:
            train_loader, test_loader = data.get_precomputed(
                '../data/'+d['dataset'], normalize=normalize)


def get_model(prefix, version='final', di=None, llr=False):
    if di is not None:
        load_data(di)

    batchnorm = d.get('batchnorm')

    if batchnorm is None:
        print('Assuming no batchnorm')
        batchnorm = False

    # localization_class = partial(
    #     models.localization_dict[d['localization']],
    #     parameters = d['localization_parameters'],
    # )

    if llr is not False:
        llr = d['loc_lr_multiplier'] if llr is True else llr
        localization_class = partial(localization_class, loc_lr_multiplier=llr)

    model = models.model_dict[d['model']](
        parameters = d['model_parameters'],
        input_shape = train_loader.dataset[0][0].shape,
        localization_class = models.localization_dict[d['localization']],
        localization_parameters = d['localization_parameters'],
        stn_placement = d['stn_placement'],
        loop = d['loop'],
        data_tag = d['dataset'],
        batchnorm = batchnorm,
    )
    
    model.load_state_dict(t.load(
        directory+'/'+str(prefix)+version,
        map_location='cpu',
    ))
    # model.load_state_dict(t.load(directory+prefix+"ckpt"+"100"))
    return model


cross_entropy_sum = t.nn.CrossEntropyLoss(reduction='sum')
def test_model(model=0, di=None, normalize=True, test_data=None, runs=1):
    global test_loader

    if type(model) == int:
        model = get_model(model, di=di)
    if test_data is not None:
        test_loader = test_data

    is_svhn = path.dirname(d['dataset'])[-4:] == 'svhn'

    losses = []
    accs = []

    with t.no_grad():
        model.eval()

        for i in range(runs):
            test_loss = 0
            correct = 0
            for x, y in test_loader:
                # x, y = x.to(device), y.to(device)
                output = model(x)

                if is_svhn:
                    loss = sum([cross_entropy_sum(output[i],y[:,i]) for i in range(5)])
                    pred = t.stack(output, 2).argmax(1)
                    correct += pred.eq(y).all(1).sum().item()
                else:
                    loss = cross_entropy_sum(output, y)
                    pred = output.argmax(1, keepdim=True)
                    correct += pred.eq(y.view_as(pred)).sum().item()
                test_loss += loss.item()
    
            test_loss /= len(test_loader.dataset)
            correct /= len(test_loader.dataset)

            losses.append(test_loss)
            accs.append(correct)

    print('Average loss:', sum(losses) / runs)
    print('Average accuracy:', sum(accs) / runs)
    return losses, accs


def running_mean(l, w):
    if w <= 0:
        return l
    r = np.zeros_like(l)
    for i,e in enumerate(l):
        n = 0
        for j in range(i-w, i+w+1):
            if 0 <= j < len(l):
                r[i] += l[j]
                n += 1
        r[i] /= n
    return r

def print_history(prefixes=[0,1,2],loss=False,start=0,di=None,window=0):
    global history
    if di is not None:
        load_data(di)
    if type(prefixes) == int:
        prefixes = [prefixes]
    for prefix in prefixes:
        history = t.load(directory + '/' + str(prefix) + "history")
        print('PREFIX', prefix)
        print('Max test_acc', np.argmax(history['test_acc']), np.max(history['test_acc']))
        print('Max train_acc', np.argmax(history['train_acc']), np.max(history['train_acc']))
        print('Final test_acc', history['test_acc'][-1])
        print('Final train_acc', history['train_acc'][-1])
        r = range(start, len(history['train_loss']))
        if loss:
            plt.plot(r, running_mean(history['train_loss'][start:], window))
            plt.plot(r, running_mean(history['test_loss'][start:], window))
        else:
            plt.plot(r, running_mean(history['train_acc'], window)[start:])
            plt.plot(r, running_mean(history['test_acc'], window)[start:])
        plt.show()

def test_multi_stn(model=0, n=4, di=None, version='final'):
    if type(model) == int:
        model = get_model(model, di=di, version=version)
    model.eval()
    batch = next(iter(test_loader))[0][:n]
    assert not d.get('batchnorm')
    if not d['loop']:
        transformed = [batch]
        x = batch
        for i,m in enumerate(model.pre_stn):
            loc_input = m(x)
            theta = model.localization[i](loc_input)
            x = model.stn(theta, loc_input)
            transformed.append(model.stn(theta, transformed[-1]))
    else:
        transformed = [batch]
        serial = [batch]
        theta = t.eye(3)
        x = batch
        for i,m in enumerate(model.loop_models):
            localization_output = model.localization[i](m(x))
            serial.append(model.stn(localization_output, serial[-1]))
            mat = F.pad(localization_output, (0,3)).view((-1,3,3))
            mat[:,2,2] = 1
            theta = t.matmul(theta,mat)
            # note that the new transformation is multiplied
            # from the right. Since the parameters are the
            # inverse of the parameters that would be applied
            # to the numbers, this yields the same parameters
            # that would result from each transformation being
            # applied after the previous, with the stn.
            # Empirically, there's no noticeable difference
            # between multiplying from the right and left.
            x = model.stn(theta[:,0:2,:], batch)
            transformed.append(x)

    minimum = t.min(batch)
    maximum = t.max(batch)

    k = len(transformed)
    f, axs = plt.subplots(2,2,figsize=(10,10))
    for j,images in enumerate(transformed):
        for i,image in enumerate(images):
            image = (image.detach() - minimum) / (maximum - minimum)
            if image.shape[0] == 1:
                image = image[0,:,:]
            else:
                image = np.moveaxis(image.numpy(),0,-1)
            plt.subplot(n,k,i*k + 1 + j)
            plt.imshow(image)
    plt.show()

    # f, axs = plt.subplots(2,2,figsize=(10,10))
    # for j,images in enumerate(serial):
    #     for i,image in enumerate(images):
    #         image = (image.detach() - minimum) / (maximum - minimum)
    #         if image.shape[0] == 1:
    #             image = image[0,:,:]
    #         else:
    #             image = np.moveaxis(image.numpy(),0,-1)
    #         plt.subplot(n,k,i*k + 1 + j)
    #         plt.imshow(image)
    # plt.show()

def test_stn(model=0, n=4, di=None, version='final'):
    if type(model) == int:
        model = get_model(model, di=di, version=version)
    model.eval()

    batch = next(iter(test_loader))[0][:n]
    theta = model.localization[0](model.pre_stn[0](batch))
    transformed_batch = model.stn(theta, batch)

    minimum = t.min(batch)
    maximum = t.max(batch)

    for image,transformed in zip(batch, transformed_batch):
        image = (image - minimum) / (maximum - minimum)
        transformed = (transformed.detach() - minimum) / (maximum - minimum)
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


def hook(x):
    print('Shape', x.shape)
    print('Median abs', t.median(t.abs(x)))
    print('Max', t.max(x), 'Min', t.min(x))
    summed = t.sum(x, dim=0)
    print('Median sum', t.median(t.abs(summed)))
    print('Max', t.max(summed), 'Min', t.min(summed))
    print('Vector length', t.norm(summed.view(-1)))

def module_hook(module, grad_input, grad_output):
    print('module', module)
    print('grad input', [i.shape for i in grad_input])
    print('grad output', [o.shape for o in grad_output])

def get_gradients(model=0, di=None, version='final'):
    assert type(model) == int
    if di is not None:
        load_data(di)

    input_image,y = next(iter(train_loader))

    if d['loop']:
        for model in [get_model(model, version, llr=llr) for llr in [False,0]]:
            model.train()
            x = input_image
            theta = t.eye(3)
            for i,m in enumerate(model.loop_models):
                loc_input = m(x)
                # if x.requires_grad:
                #     loc_input.register_hook(lambda a: print('loc input', hook(a), '\n'))
                # model.localization[i].register_backward_hook(module_hook)
                loc_output = model.localization[i](loc_input)
                # loc_output.register_hook(lambda a: print('loc output', hook(a), '\n'))
                mat = F.pad(loc_output, (0,3)).view((-1,3,3))
                mat[:,2,2] = 1
                theta = t.matmul(theta,mat)
                x = model.stn(theta[:,0:2,:], input_image)
                # x.register_hook(lambda a: print('stn out', hook(a),'\n'))
            x = m(x)
            # x.register_hook(lambda a: print('final m', hook(a),'\n'))
            x = model.final_layers(x)
            # x.register_hook(lambda a: print('final layers', hook(a),'\n'))
            output = model.output(x.view(x.size(0),-1))
            # [out.register_hook(lambda a: print('output', hook(a),'\n')) for out in output]

            assert path.dirname(d['dataset'])[-4:] == 'svhn'
            loss = sum([F.cross_entropy(output[i],y[:,i]) for i in range(5)])
            loss.backward()

            print('\nGradients')
            for i,(l,m) in enumerate(zip(model.localization, model.pre_stn)):
                print('Pre stn', i)
                for p in m.parameters():
                    print(p.grad.shape)
                    print(t.norm(p.grad.view(-1)))
                    print()

                print('Localization', i)
                for p in l.parameters():
                    print(p.grad.shape)
                    print(t.norm(p.grad.view(-1)))
                    print()

            print('Final layers')
            for p in model.final_layers.parameters():
                print(p.grad.shape)
                print(t.norm(p.grad.view(-1)))
                print()
    else:
        model = get_model(model, version)
        model.train()
        output = model(input_image)
        loss = sum([F.cross_entropy(output[i],y[:,i]) for i in range(5)])
        loss.backward()

        for i,(l,m) in enumerate(zip(model.localization, model.pre_stn)):
            print('Pre stn', i)
            for p in m.parameters():
                print(p.grad.shape)
                print(t.norm(p.grad.view(-1)))
                print()

            print('Localization', i)
            for p in l.parameters():
                print(p.grad.shape)
                print(t.norm(p.grad.view(-1)))
                print()

        print('Final layers')
        for p in model.final_layers.parameters():
            print(p.grad.shape)
            print(t.norm(p.grad.view(-1)))
            print()




def get_rotated_images(model=0, di=None, normalization=True,
                       tall=False, save_path='', title=''):
    if type(model) == int:
        model = get_model(model, di=di)

    model.eval()

    # _, unrotated_test = data.data_dict[d['dataset']](
    #     rotate=False, normalize=False) 
    images = t.zeros(10,1,28,28)
    numbers = set()
    for x,y in untransformed_test:
        for im,l in zip(x,y):
            if l not in numbers:
                numbers.add(l)
                images[l] = im
        if len(numbers) == 10:
            break

    angles = np.random.uniform(-90, 90, 3*10)
    rot_x = t.tensor([
        rotate(images[i // 3][0], angle) for i, angle in enumerate(angles)
    ], dtype=t.float).reshape(-1, 1, 28, 28)
    if normalization:
        rot_x = (rot_x - 0.1307) / 0.3081

    # bordered_rot_x = copy.deepcopy(rot_x)
    # bordered_rot_x = bordered_rot_x * 0.3081 + 0.1307 # normalization
    # for i in range(28):
    #     bordered_rot_x[:,0,i,0] = 1
    #     bordered_rot_x[:,0,0,i] = 1
    #     bordered_rot_x[:,0,-1,i] = 1
    #     bordered_rot_x[:,0,i,-1] = 1

    stn_x = model.stn(model.localization[0](model.pre_stn[0](rot_x)), rot_x)
    # only handles models with a single stn

    if tall:
        fig, axs = plt.subplots(10, 6, sharex='col', sharey='row', figsize=(6,10),
                                gridspec_kw={'hspace': 0.02, 'wspace': 0.02})
        for i in range(6):
            for j in range(0, 10, 2):
                axs[j, i].imshow(rot_x[i + 3*j].detach().numpy()[0])
                axs[j+1, i].imshow(stn_x[i + 3*j].detach().numpy()[0])
                axs[j,i].axis(False)
                axs[j+1,i].axis(False)

    else:
        fig, axs = plt.subplots(6, 10, sharex='col', sharey='row', figsize=(10,6),
                                gridspec_kw={'hspace': 0.02, 'wspace': 0.02})
        for i in range(6):
            for j in range(0, 10, 2):
                axs[i, j].imshow(rot_x[i + 3*j].detach().numpy()[0])
                axs[i, j+1].imshow(stn_x[i + 3*j].detach().numpy()[0])
                axs[i,j].axis(False)
                axs[i,j+1].axis(False)

    # for col, (r, s) in enumerate(zip(rot_x, stn_x)):
        # axs[0, col].imshow(r.detach().numpy()[0])
        # axs[1, col].imshow(s.detach().numpy()[0])

    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.show()

def compare_rotations(di1, di2, model1=0, model2=0, angles=[], normalization=True,
                      ylabels = ['','',''], save_path='', title=''):
    model = get_model(model1, di=di1)
    batch = next(iter(untransformed_test))[0][:1]

    if len(angles) == 0:
        angles = np.random.uniform(-90, 90, 3)
    assert len(angles) == 3

    rot_x = t.tensor([
        rotate(batch[i // 3][0], angle) for i, angle in enumerate(angles)
    ], dtype=t.float).reshape(-1, 1, 28, 28)
    if normalization:
        rot_x = (rot_x - 0.1307) / 0.3081

    model.eval()
    theta = model.localization[0](model.pre_stn[0](rot_x))
    stn1 = model.stn(theta, rot_x)
  
    model = get_model(model2, di=di2)
    model.eval()
    theta = model.localization[0](model.pre_stn[0](rot_x))
    stn2 = model.stn(theta, rot_x)

    fig, axs = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(3,3.04),
                            gridspec_kw={'hspace': 0.02, 'wspace': 0.02})
    plt.gray()
    for i in range(3):
        axs[0,i].imshow(rot_x[i].detach().numpy()[0])
        axs[1,i].imshow(stn1[i].detach().numpy()[0])
        axs[2,i].imshow(stn2[i].detach().numpy()[0])
        axs[0,i].set_xticks([])
        axs[0,i].set_yticks([])
        axs[1,i].set_xticks([])
        axs[1,i].set_yticks([])
        axs[2,i].set_xticks([])
        axs[2,i].set_yticks([])

    for ax,y in zip(axs[:,0], ylabels):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(y)
    
    fig.suptitle(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()


def compare_stns(di1, di2, model1=0, model2=0, save_path='', title=''):
    n = 5

    model = get_model(model1, di=di1)
    batch = next(iter(test_loader))[0][:n]

    model.eval()
    theta = model.localization[0](model.pre_stn[0](batch))
    stn1 = model.stn(theta, batch)

    model = get_model(model2, di=di2)
    model.eval()
    theta = model.localization[0](model.pre_stn[0](batch))
    stn2 = model.stn(theta, batch)

    plt.gray()
    fig, axs = plt.subplots(3, n, sharex='col', sharey='row', figsize=(n,3),
                            gridspec_kw={'hspace': 0.02, 'wspace': 0.02})
    minimum = t.min(batch)
    maximum = t.max(batch)
    batch = (batch.detach() - minimum) / (maximum - minimum)
    stn1 = (stn1.detach() - minimum) / (maximum - minimum)
    stn2 = (stn2.detach() - minimum) / (maximum - minimum)
    for i in range(n):
        axs[0,i].imshow(np.moveaxis(batch[i].numpy(), 0, -1))
        axs[1,i].imshow(np.moveaxis(stn1[i].numpy(), 0, -1))
        axs[2,i].imshow(np.moveaxis(stn2[i].numpy(), 0, -1))
        axs[0,i].axis(False)
        axs[1,i].axis(False)
        axs[2,i].axis(False)

    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.show()


def angle_from_matrix(thetas, all_transformations=False):
    # V1: Inverts in order to get parameters for the number's
    #     transformation, and decomposes into Scale Shear Rot
    mat = F.pad(thetas, (0, 3)).view(-1,3,3)
    mat[:,2,2] = 1
    transform = t.inverse(mat)
    angle = (np.arctan2(transform[:,0,1], transform[:,0,0])) * 180 / np.pi
    # negated twice because the y-axis is inverted and 
    # because I use counter-clockwise as positive direction
    if not all_transformations:
        return angle

    det = transform[:,0,0]*transform[:,1,1] - transform[:,0,1]*transform[:,1,0]
    shear = (transform[:,0,0]*transform[:,1,0] + transform[:,0,1]*transform[:,1,1] / det)
    scale_x = np.sqrt(transform[:,0,0]**2 + transform[:,0,1]**2)
    scale_y = det / scale_x
    return angle, shear, scale_x, scale_y, det

    # # V2: Decomposes the window's transformation into Scale Shear Rot.
    # #     This Rot*-1 is equal to the inverse's decomposed into Rot Shear Scale.
    # thetas = thetas.view(-1,2,3)
    # return -(np.arctan(thetas[:,0,1] / thetas[:,0,0])) * 180 / np.pi
    # # negated because the images is transformed in the reverse
    # # of the predicted transform, because the y-axis is inverted,
    # # and because I use counter-clockwise as positive direction


def plot_angles(rot, pred, line='equation', save_path='', title='',
                xlabel='', ylabel='', pointlabel='', ll='best'):
    plt.figure(figsize=(3,3))
    heatmap, xedges, yedges = np.histogram2d(
        rot, pred, bins=110, range=[[-110,110],[-110,110]])
    extent = [-110, 110, -110, 110]
    plt.imshow(heatmap.T, extent=extent, cmap='Greys', origin = 'lower')
    plt.xticks([-90,-45,0,45,90])
    plt.yticks([-90,-45,0,45,90])

    plt.scatter([-1],[-1],s=2,c='black',label=pointlabel)

    if line:
        # # Minimize vertical error
        # A = np.vstack([rot, np.ones(len(rot))]).T
        # m,c = np.linalg.lstsq(A, pred, rcond=None)[0]

        # Minimize orthogonal error (https://en.wikipedia.org/wiki/Deming_regression)
        n = len(rot)
        x = np.mean(rot)
        y = np.mean(pred)
        sxx = sum((rot-x)**2)/(n-1)
        sxy = sum((rot-x)*(pred-y))/(n-1)
        syy = sum((pred-y)**2)/(n-1)
        m = (syy - 1*sxx + np.sqrt((syy - 1*sxx)**2 + 4*1*sxy**2))/(2*sxy)
        c = y - m*x

        print('m:', m, '  c:', c)
        if line == 'equation':
            label = '{:.2f}x {} {:.1f}'.format(m, '-' if c<0 else '+', abs(c))
        elif line is not True:
            label = line
        else:
            label = ''
        plt.plot(rot, m*rot + c, 'r', label=label)

    plt.legend(loc=ll)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(save_path, bbox_inches = 'tight', pad_inches=.03)
    plt.show()

def rotation_statistics(model=0, plot='all', di=None, all_transformations=False,
                        normalize=True, epochs=1, save_path='', title=''):
    global unrotated_test
    if type(model) == int:
        model = get_model(model, di=di)
        _, unrotated_test = data.data_dict[d['dataset']](
            rotate=False, normalize=False)
    elif 'unrotated_test' not in globals():
        _, unrotated_test = data.data_dict[d['dataset']](
            rotate=False, normalize=False)

    device = t.device("cuda" if next(model.parameters()).is_cuda else "cpu")

    rotated_angles = np.array([])
    predicted_angles = np.array([])
    rot_by_label = []
    pred_by_label = []

    labels = np.array([])
    
    if all_transformations:
        shears = np.array([])
        scale_xs = np.array([])
        scale_ys = np.array([])
        dets = np.array([])
        shear_by_label = []
        sx_by_label = []
        sy_by_label = []
        det_by_label = []

    with t.no_grad():
        model.eval()

        for epoch in range(epochs):
            for x, y in unrotated_test:
                angles = np.random.uniform(-90, 90, x.shape[0])
                rot_x = t.tensor([
                    rotate(im[0], angle) for im, angle in zip(x, angles)
                ], dtype=t.float).reshape(-1, 1, 28, 28)
                if normalize:
                    rot_x = (rot_x - 0.1307) / 0.3081

                theta = model.localization[0](model.pre_stn[0](rot_x.to(device))).cpu()

                rotated_angles = np.append(rotated_angles, angles)
                labels = np.append(labels, y)

                if all_transformations:
                    angle, shear, sx, sy, det = angle_from_matrix(theta, all_transformations=True)
                    predicted_angles = np.append(predicted_angles, angle)
                    shears = np.append(shears, shear)
                    scale_xs = np.append(scale_xs, sx)
                    scale_ys = np.append(scale_ys, sy)
                    dets = np.append(dets, det)
                else:
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

    variance = 0
    for i in range(10):
        indices = labels==i
        rot_by_label.append(rotated_angles[indices])
        pred_by_label.append(predicted_angles[indices])

        s = (sum(rot_by_label[i]) + sum(pred_by_label[i]))/len(rot_by_label[i])
        variance += sum([(r+p - s)**2 for r,p in zip(rot_by_label[i],pred_by_label[i])])

        if all_transformations:
            shear_by_label.append(shears[indices])
            sx_by_label.append(scale_xs[indices])
            sy_by_label.append(scale_ys[indices])
            det_by_label.append(dets[indices])

    print('Standard deviation', np.sqrt(variance / (epochs * 10000)))

    if plot == 'sep': # plot all digits separately
        assert not save_path, "Haven't implemented saving of more than one image"
        for i in range(10):
            print('Plotting label', i)
            plot_angles(rot_by_label[i], pred_by_label[i])

            if all_transformations:
                plot_angles(rot_by_label[i], 100 * shear_by_label[i])
                plot_angles(rot_by_label[i], 50 * sx_by_label[i])
                plot_angles(rot_by_label[i], 50 * sy_by_label[i])
                # plt.hist(shear_by_label[i], 50, range=(-1.25, 1.25))
                # plt.figure()
                # plt.hist(sx_by_label[i], 50, range=(0.5, 2.5))
                # plt.figure()
                # plt.hist(sy_by_label[i], 50, range=(0.5, 2.5))
                # plt.show()

    if plot == 'all': # plot all data at once
        plot_angles(rotated_angles, predicted_angles, save_path=save_path, title=title)
        
        if all_transformations:
            assert not save_path, "Haven't implemented saving of more than one image"
            plot_angles(rotated_angles, 100 * shears)
            plot_angles(rotated_angles, 50 * scale_xs)
            plot_angles(rotated_angles, 50 * scale_ys)


    if all_transformations:
        return rot_by_label, pred_by_label, shear_by_label, sx_by_label, sy_by_label, det_by_label
    return rot_by_label, pred_by_label


def distance_from_matrix(thetas):
    thetas = thetas.view((-1,2,3))
    # distances = np.array([
    #     np.linalg.solve(theta[:,0:2], theta[:,2]) for theta in thetas
    # ])
    return np.array(thetas[:,:,2]) * np.array([-1, 1]) * 30
    # This is probably wrong for translations beyond the third layer.
    # Is negated twice because the digits are transformed in the reverse
    # of predicted transform, and because the y-axis is inverted.

def plot_distance(tran, pred):
    heatmap, xedges, yedges = np.histogram2d(
        tran[:,0], pred[:,0], bins=32, range=[[-16,16],[-16,16]])
    extent = [-16, 16, -16, 16]
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.figure()

    heatmap, xedges, yedges = np.histogram2d(
        tran[:,1], pred[:,1], bins=32, range=[[-16,16],[-16,16]])
    extent = [-16, 16, -16, 16]
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()

def translation_statistics(model=0, plot=True, di=None, all_transformations=False):
    global untranslated_test
    if type(model) == int:
        model = get_model(model, di=di)
        _, untranslated_test = data.mnist(rotate=False, normalize=False, translate=False)
    elif 'untranslated_test' not in globals():
        _, untranslated_test = data.mnist(rotate=False, normalize=False, translate=False)

    device = t.device("cuda" if next(model.parameters()).is_cuda else "cpu")

    noise = data.MNIST_noise()

    translated_distance = np.zeros((0,2))
    predicted_distance = np.zeros((0,2))
    tran_by_label = []
    pred_by_label = []

    labels = np.array([])

    if all_transformations:
        angles = np.array([])
        shears = np.array([])
        scale_xs = np.array([])
        scale_ys = np.array([])
        dets = np.array([])
        angle_by_label = []
        shear_by_label = []
        sx_by_label = []
        sy_by_label = []
        det_by_label = []

    with t.no_grad():
        model.eval()

        for x, y in untranslated_test:
            distance = np.random.randint(-16, 17, (x.shape[0], 2))
            translated = t.zeros(x.shape[0], 1, 60, 60, dtype=t.float)
            for i,(im,(xd,yd)) in enumerate(zip(x, distance)):
                translated[i, 0, 16-yd : 44-yd, 16+xd : 44+xd] = im[0]
                translated[i] = noise(translated[i])

            if 'normalize' not in d:
                print('Assuming no normalization.')
            elif d['normalize']:
                translated = (translated - 0.0363) / 0.1870
            
            theta = model.localization[0](model.pre_stn[0](translated.to(device))).cpu()

            translated_distance = np.append(translated_distance, distance, axis=0)
            predicted_distance = np.append(
                predicted_distance, distance_from_matrix(theta), axis=0)

            labels = np.append(labels, y)

            if all_transformations:
                angle, shear, sx, sy, det = angle_from_matrix(theta, all_transformations=True)
                angles = np.append(angles, angle)
                shears = np.append(shears, shear)
                scale_xs = np.append(scale_xs, sx)
                scale_ys = np.append(scale_ys, sy)
                dets = np.append(dets, det)

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

    variance = 0
    for i in range(10):
        indices = labels==i
        tran_by_label.append(translated_distance[indices,:])
        pred_by_label.append(predicted_distance[indices,:])

        s = (sum(tran_by_label[i]) + sum(pred_by_label[i]))/len(tran_by_label[i])
        variance += sum([np.linalg.norm(t+p - s) for t,p in zip(tran_by_label[i],pred_by_label[i])])

        if all_transformations:
            angle_by_label.append(angles[indices])
            shear_by_label.append(shears[indices])
            sx_by_label.append(scale_xs[indices])
            sy_by_label.append(scale_ys[indices])
            det_by_label.append(dets[indices])

    print('Standard deviation', np.sqrt(variance / 10000))

    if plot:
        plot_distance(translated_distance, predicted_distance)

    if all_transformations:
        return tran_by_label, pred_by_label, angle_by_label, shear_by_label, sx_by_label, sy_by_label, det_by_label
    return tran_by_label, pred_by_label

def average_n(res, n):
    for run in res:
        s = 0
        for label in run[n]:
            s += sum(label)
        s /= len(untransformed_test.dataset)
        print(s)

def compare_translation(di1, di2, model1=0, model2=0, angles=[], normalization=True,
                        ylabels = ['','',''], save_path='', title=''):
    load_data(di1)
    im = next(iter(untransformed_test))[0][:1]

    noise = data.MNIST_noise()
    distance = np.random.randint(-16, 17, (3, 2))
    translated = t.zeros(3, 1, 60, 60, dtype=t.float)
    for i, (xd, yd) in enumerate(distance):
        translated[i, 0, 16-yd : 44-yd, 16+xd : 44+xd] = im[0]
        translated[i] = noise(translated[i])

    if 'normalize' not in d:
        print('Assuming no normalization.')
    elif d['normalize']:
        translated = (translated - 0.0363) / 0.1870
            
    model = get_model(model1)
    model.eval()
    theta = model.localization[0](model.pre_stn[0](translated))
    stn1 = model.stn(theta, translated)
  
    load_data(di2)
    model = get_model(model2)
    model.eval()
    theta = model.localization[0](model.pre_stn[0](translated))
    stn2 = model.stn(theta, translated)

    fig, axs = plt.subplots(3, 3,  figsize=(3,3.04), # sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0.02, 'wspace': 0.02})
    plt.gray()
    for i in range(3):
        axs[0,i].imshow(translated[i].detach().numpy()[0])
        axs[1,i].imshow(stn1[i].detach().numpy()[0])
        axs[2,i].imshow(stn2[i].detach().numpy()[0])
        axs[0,i].set_xticks([])
        axs[0,i].set_yticks([])
        axs[1,i].set_xticks([])
        axs[1,i].set_yticks([])
        axs[2,i].set_xticks([])
        axs[2,i].set_yticks([])

    for ax,y in zip(axs[:,0], ylabels):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(y)
    
    fig.suptitle(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()


def plot_results(folder, n_prefixes, *args):
    for l in args:
        accs = []
        for d in l:
            accs.append([])
            for prefix in range(n_prefixes):
                history = t.load(folder + d + '/' + str(prefix) + "history")
                accs[-1].append(history['test_acc'][-1])
            accs[-1].sort()
            accs[-1] = accs[-1][n_prefixes // 2]
        plt.plot(accs)
    plt.show()
