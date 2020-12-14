#%%
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import PIL.Image
import numpy as np
import models
import data
import angles
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from skimage.transform import rotate
from os import path
from functools import partial
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

directory, d, train_loader, test_loader  = None, None, None, None

#%% Functions

### LOAD DATA AND MODELS ###

def load_data(data_dir, normalize=True):
    global directory, d, train_loader, test_loader, untransformed_test
    directory = '../experiments/'+data_dir

    d = torch.load(directory+"/model_details")
    if 'normalize' in d:
        normalize = d['normalize']
    elif d['dataset'] == 'translate':
        print('Assuming no normalizaion on translated data')
        normalize = False
    
    if 'deep-localization' in data_dir:
        print('In deep-localization; setting deep=True')
        d['deep'] = True

    if d['dataset'] in data.data_dict:
        train_loader, test_loader = data.data_dict[d['dataset']](
            rotate = d['rotate'], normalize = normalize)
        if d['dataset'] in ['mnist', 'translate', 'scale']:
            _, untransformed_test = data.mnist(rotate=False, translate=False, normalize=False)
    else:
        try:
            train_loader, test_loader = data.get_precomputed(
                '../'+d['dataset'], normalize=normalize)
        except FileNotFoundError:
            try:
                train_loader, test_loader = data.get_precomputed(
                    '../data/'+d['dataset'], normalize=normalize)
            except FileNotFoundError:
                train_loader, test_loader = data.get_precomputed(
                    d['dataset'], normalize=normalize)
        untransformed_test = test_loader


def get_model(prefix, version='final', di=None, llr=False, add_iterations=0):
    if di is not None:
        load_data(di)

    

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
        batchnorm = d.get('batchnorm', False),
        deep = d.get('deep', False),
        # mean = train_loader.dataset.mean
    )
    for i in d.get('add_iteration', []):
        model.add_iteration()
    
    unexpected = model.load_state_dict(
        torch.load(
            directory+'/'+str(prefix)+version,
            map_location='cpu'),
        strict = False,
    )
    if unexpected.missing_keys or unexpected.unexpected_keys:
        print('State dict did not match model. Unexpected:', unexpected)
    # model.load_state_dict(torch.load(directory+prefix+"ckpt"+"100"))
    for i in range(add_iterations):
        model.add_iteration()

    return model


### COMPUTE AND PRINT ACCURACY AND LOSS ###

cross_entropy_sum = torch.nn.CrossEntropyLoss(reduction='sum')
def test_model(model=0, di=None, test_data=None, runs=1):
    global test_loader

    if type(model) == int:
        model = get_model(model, di=di)
    if test_data is not None:
        test_loader = test_data

    is_svhn = path.dirname(d['dataset'])[-4:] == 'svhn'

    losses = []
    accs = []
    true_labels = np.array([])
    predicted_labels = np.array([])

    with torch.no_grad():
        model.eval()

        for i in range(runs):
            test_loss = 0
            correct = 0
            for i,(x,y) in enumerate(test_loader):
                # x, y = x.to(device), y.to(device)
                output = model(x)

                if is_svhn:
                    loss = sum([cross_entropy_sum(output[i],y[:,i]) for i in range(5)])
                    pred = torch.stack(output, 2).argmax(1)
                    correct += pred.eq(y).all(1).sum().item()
                else:
                    loss = cross_entropy_sum(output, y)
                    pred = output.argmax(1, keepdim=True)
                    correct += pred.eq(y.view_as(pred)).sum().item()
                test_loss += loss.item()

                true_labels = np.concatenate((true_labels, y.reshape(-1)))
                predicted_labels = np.concatenate((predicted_labels, pred.reshape(-1)))

                print(correct/(len(x)*(1+i)), end=' ', flush=True)
    
            test_loss /= len(test_loader.dataset)
            correct /= len(test_loader.dataset)

            losses.append(test_loss)
            accs.append(correct)

    print('Average loss:', sum(losses) / runs)
    print('Average accuracy:', sum(accs) / runs)
    return true_labels, predicted_labels

def get_confusion_matrix(model=0, di=None, true_labels=None, predicted_labels=None):
    if true_labels is None:
        true_labels, predicted_labels = test_model(model, di)

    sns.set_style('whitegrid', {'axes.grid': False})
    cmap = plt.cm.Blues

    fig, ax = plt.subplots(1,1, figsize=[4,4])

    cm = confusion_matrix(true_labels, predicted_labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cax = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1.)
    ax.grid(False)
    plt.colorbar(cax, ax=ax)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.show()

    return true_labels, predicted_labels

def accuracy_v_nexamples(model=0, di=None, true_labels=None, predicted_labels=None):
    if true_labels is None:
        true_labels, predicted_labels = test_model(model, di)

    nexamples = np.bincount(true_labels.astype(int))
    cm = confusion_matrix(true_labels, predicted_labels)
    falsepositives_n = cm.sum(axis=0) - [cm[i,i] for i in range(len(cm))]
    falsepositives_r = falsepositives_n / cm.sum(axis=0)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accuracy = [cm[i,i] for i in range(len(nexamples))]

    plt.scatter(nexamples, accuracy, s=8)
    plt.scatter(nexamples, falsepositives_r, s=8)
    plt.figure()
    plt.scatter(np.log2(nexamples), accuracy, s=8)
    plt.scatter(np.log2(nexamples), falsepositives_r, s=8)
    plt.figure()
    plt.scatter(nexamples, falsepositives_n, s=8)
    plt.figure()
    plt.scatter(np.log2(nexamples), falsepositives_n, s=8)
    plt.show()


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

def print_history(prefixes=[0,1,2],loss=False,start=0,di=None,window=0,show=True):
    global history
    if di is not None:
        load_data(di)
    if type(prefixes) == int:
        prefixes = [prefixes]
    for prefix in prefixes:
        history = torch.load(directory + '/' + str(prefix) + "history")
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
        if show:
            plt.show()


### VIEW TRANSFORMED IMAGES ###

def test_stn(model=0, n=4, di=None, version='final'):
    if type(model) == int:
        model = get_model(model, di=di, version=version)
    model.eval()

    batch = next(iter(test_loader))[0][:n]
    theta = model.localization[0](model.pre_stn[0](batch))
    transformed_batch = model.stn(theta, batch)

    minimum = torch.min(batch)
    maximum = torch.max(batch)

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

def test_multi_stn(model=0, n=4, di=None, version='final', param=[]):
    if type(model) == int:
        model = get_model(model, di=di, version=version)
    model.eval()
    try:
        test_loader.dataset.set_moment_probability(0)
    except:
        print('no moment prob')
    batch = next(iter(test_loader))[0][:n]
    if len(param) == n:
        if d['normalize']:
            normalizer = test_loader.dataset.transform.transforms[-1]
            batch = batch * normalizer.std[0] + normalizer.mean[0]
        batch = torch.tensor([
            rotate(batch[0][0], angle) for angle in param
        ], dtype=torch.float).reshape(batch.shape)
        if d['normalize']:
            batch = (batch - normalizer.mean[0]) / normalizer.std[0]

    if not d['loop']:
        assert not d.get('batchnorm')
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
        theta = torch.eye(3)
        x = batch
        for i,m in enumerate(model.loop_models):
            localization_output = model.localization[i](m(x))
            serial.append(model.stn(localization_output, serial[-1]))
            mat = F.pad(localization_output, (0,3)).view((-1,3,3))
            mat[:,2,2] = 1
            theta = torch.matmul(theta,mat)
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
            if d.get('batchnorm'):
                x = model.batchnorm[i](x)

    minimum = torch.min(batch)
    maximum = torch.max(batch)

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


def compare_all_labels(model=0, di=None, normalization=True,
                       tall=False, save_path='', title=''):
    if type(model) == int:
        model = get_model(model, di=di)
    model.eval()

    # _, unrotated_test = data.data_dict[d['dataset']](
    #     rotate=False, normalize=False) 
    images = torch.zeros(10,1,28,28)
    numbers = set()
    for x,y in untransformed_test:
        for im,l in zip(x,y):
            if l not in numbers:
                numbers.add(l)
                images[l] = im
        if len(numbers) == 10:
            break

    angles = np.random.uniform(-90, 90, 3*10)
    rot_x = torch.tensor([
        rotate(images[i // 3][0], angle) for i, angle in enumerate(angles)
    ], dtype=torch.float).reshape(-1, 1, 28, 28)
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


def compare_stns(di1, di2, model1=0, model2=0, im_nums=None, n=5,
                 save_path='', title='', ylabels=[]):
    model = get_model(model1, di=di1)

    if im_nums is None:
        im_nums = np.random.randint(0,len(test_loader.dataset),size=n)
        print('im nums: ', im_nums)
    n = len(im_nums)
    batch = [test_loader.dataset[im_num][0] for im_num in im_nums]
    batch = torch.stack(batch)

    model.eval()
    theta = model.localization[0](model.pre_stn[0](batch))
    stn1 = model.stn(theta, batch)

    model = get_model(model2, di=di2)
    model.eval()
    theta = model.localization[0](model.pre_stn[0](batch))
    stn2 = model.stn(theta, batch)

    # plt.gray()
    fig, axs = plt.subplots(3, n, sharex='col', sharey='row', figsize=(n,3),
                            gridspec_kw={'hspace': 0.02, 'wspace': 0.02}, frameon=False)
    minimum = torch.min(batch)
    maximum = torch.max(batch)
    batch = (batch.detach() - minimum) / (maximum - minimum)
    stn1 = (stn1.detach() - minimum) / (maximum - minimum)
    stn2 = (stn2.detach() - minimum) / (maximum - minimum)
    for i in range(n):
        axs[0,i].imshow(np.moveaxis(batch[i].numpy(), 0, -1))
        axs[1,i].imshow(np.moveaxis(stn1[i].numpy(), 0, -1))
        axs[2,i].imshow(np.moveaxis(stn2[i].numpy(), 0, -1))
        axs[0,i].set_xticks([])
        axs[0,i].set_yticks([])
        axs[0,i].set_frame_on(False)
        axs[1,i].set_xticks([])
        axs[1,i].set_yticks([])
        axs[1,i].set_frame_on(False)
        axs[2,i].set_xticks([])
        axs[2,i].set_yticks([])
        axs[2,i].set_frame_on(False)

    for ax,y in zip(axs[:,0], ylabels):
        ax.set_ylabel(y,fontsize=12)

    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.show()

def compare_plankton_transformation(model=0, di=None, param=[], normalize=None):
    if di is not None:
        model = get_model(model, di=di)

    test_loader.dataset.set_moment_probability(0)
    im = next(iter(test_loader))[0][:1]
    if d['normalize']:
        normalizer = test_loader.dataset.transform.transforms[-1]
        im = im * normalizer.std[0] + normalizer.mean[0]

    if len(param) == 0:
        param = np.random.uniform(-180,180,3)
    transformed = torch.tensor([
        rotate(im[0][0], angle) for angle in param
    ], dtype=torch.float).reshape(-1,1,im.size(2),im.size(3))
    if d['normalize']:
        transformed = (transformed - normalizer.mean[0]) / normalizer.std[0]

    model.eval()
    x = transformed
    theta = model.localization[0](model.pre_stn[0](x))
    stn1 = model.stn(theta, transformed)
    print(-angles.get_moment_angle(transformed))
    print(angles.angle_from_matrix(theta))

    fig, axs = plt.subplots(2, 3, figsize=(6,4), # sharex, sharey not necessary
                gridspec_kw={'hspace': 0.02, 'wspace': 0.02})
    plt.gray()
    for i in range(3):
        axs[0,i].imshow(transformed[i].detach().numpy()[0])
        axs[1,i].imshow(stn1[i].detach().numpy()[0])
        axs[0,i].set_xticks([])
        axs[0,i].set_yticks([])
        axs[1,i].set_xticks([])
        axs[1,i].set_yticks([])

    for ax in axs[:,0]:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()


def compare_transformation(di1, di2, di3=None, model1=0, model2=0, model3=0, transform='rotate', param=[],
                           normalize=None, ylabels=['','',''], save_path='', title='',
                           subtract_identity=False, im_num=None):
    assert transform in ['rotate','translate','scale']
    model = get_model(model1, di=di1)
    
    if im_num is None:
        im = next(iter(untransformed_test))[0][:1]
    else:
        im = untransformed_test.dataset[im_num][0]
        im = im.reshape(1,1,im.shape[-2],im.shape[-1])

    if transform == 'rotate':
        if len(param) == 0:
            param = np.random.uniform(-90,90,3)
        transformed = torch.tensor([
            rotate(im[0][0], angle) for angle in param
        ], dtype=torch.float).reshape(-1,1,im.size(2),im.size(3))
        if normalize or normalize is None:
            transformed = (transformed - 0.1307) / 0.3081
    elif transform == 'translate':
        if len(param) == 0:
            param = np.random.randint(-16,17,(3,2))
        noise = data.MNIST_noise(60)
        transformed = torch.zeros(3, 1, 60, 60, dtype=torch.float)
        for i, (xd, yd) in enumerate(param):
            transformed[i, 0, 16-yd : 44-yd, 16+xd : 44+xd] = im[0]
            transformed[i] = noise(transformed[i])
        if normalize or (normalize is None and d.get('normalize')):
            transformed = (transformed - 0.0363) / 0.1870
    elif transform == 'scale':
        if len(param) == 0:
            param = np.random.uniform(-1,2,3)
        scale = np.power(2, param)
        noise = data.MNIST_noise(112, scale=True)
        transformed = F.pad(im, (42,42,42,42)).repeat(3,1,1,1)
        for i,s in enumerate(scale):
            transformed[i] = tvF.to_tensor(tvF.affine(
                tvF.to_pil_image(transformed[i]),
                angle=0, translate=(0,0), shear=0, scale = s,
                resample = PIL.Image.BILINEAR, fillcolor = 0))
            transformed[i] = noise(transformed[i])
        if d['normalize']:
            transformed = (transformed - 0.0414) / 0.1751

    model.eval()
    theta = model.localization[0](model.pre_stn[0](transformed))
    theta = theta - model.localization[0].identity if subtract_identity else theta
    if len(model.pre_stn[0]) and not d['loop']:
        print('changing translate')
        theta = theta.view(-1,2,3)
        theta[:,:,2] *= 25/29.5
    stn1 = model.stn(theta, transformed)

    model = get_model(model2, di=di2)
    model.eval()
    theta = model.localization[0](model.pre_stn[0](transformed))
    theta = theta - model.localization[0].identity if subtract_identity else theta
    if len(model.pre_stn[0]) and not d['loop']:
        print('changing translate')
        theta = theta.view(-1,2,3)
        theta[:,:,2] *= 25/29.5
    stn2 = model.stn(theta, transformed)

    if di3 is not None:
        model = get_model(model3, di=di3)
        model.eval()
        theta = model.localization[0](model.pre_stn[0](transformed))
        theta = theta - model.localization[0].identity if subtract_identity else theta
        if len(model.pre_stn[0]) and not d['loop']:
            print('changing translate')
            theta = theta.view(-1,2,3)
            theta[:,:,2] *= 25/29.5
        stn3 = model.stn(theta, transformed)


    if di3 is not None:
        fig, axs = plt.subplots(4, 3, figsize=(3,4.04), # sharex, sharey not necessary
                    gridspec_kw={'hspace': 0.02, 'wspace': 0.02})
    else:
        fig, axs = plt.subplots(3, 3, figsize=(3,3.04), # sharex, sharey not necessary
                    gridspec_kw={'hspace': 0.02, 'wspace': 0.02})
    plt.gray()
    for i in range(3):
        axs[0,i].imshow(transformed[i].detach().numpy()[0])
        axs[1,i].imshow(stn1[i].detach().numpy()[0])
        axs[2,i].imshow(stn2[i].detach().numpy()[0])
        if di3 is not None:
            axs[3,i].imshow(stn3[i].detach().numpy()[0])
            axs[3,i].set_xticks([])
            axs[3,i].set_yticks([])
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


### TRANSFORMATION STATISTICS ###
def angle_from_matrix(thetas, all_transformations=False, newangle=True):

    # V1: Inverts in order to get parameters for the number's
    #     transformation, and decomposes into Scale Shear Rot
    mat = F.pad(thetas, (0, 3)).view(-1,3,3)
    mat[:,2,2] = 1

    transform = torch.inverse(mat)

    if newangle:
        scale = np.sqrt(mat[:,0,0]**2+mat[:,0,1]**2-2*mat[:,0,1]*mat[:,1,0]+
                        mat[:,1,1]**2+mat[:,1,0]**2+2*mat[:,0,0]*mat[:,1,1])/2
        angle = np.arctan2(mat[:,0,1] - mat[:,1,0], mat[:,0,0] + mat[:,1,1])
        angle *= -180 / np.pi
    else:
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


def distance_from_matrix(thetas, mp=False):
    thetas = thetas.view((-1,2,3))
    # distance = np.array([
    #     np.linalg.solve(theta[:,0:2], theta[:,2]) for theta in thetas
    # ]) * np.array([-1,1])
    distance = np.array(thetas[:,:,2]) * np.array([-1, 1])
    return distance * (25 if mp else (60-1)/2)
    # Second variable is negated twice because the digits are transformed in
    # the reverse of predicted transform, and because the y-axis is inverted.

def plot_angles(rot=None, pred=None, res=None, line='equation', save_path='', title='',
                xlabel='', ylabel='', pointlabel='', ll='best', negative=False):
    if res is not None:
        assert rot is None and pred is None
        rot = np.concatenate(res[0])
        pred = np.concatenate(res[1])
        if negative:
            pred *= -1
    plt.figure(figsize=(3,3))
    heatmap, xedges, yedges = np.histogram2d(
        rot, pred, bins=110, range=[[-110,110],[-110,110]])
    extent = [-110, 110, -110, 110]
    plt.imshow(heatmap.T, extent=extent, cmap='Greys', origin = 'lower')
    plt.xticks([-90,-45,0,45,90])
    plt.yticks([-90,-45,0,45,90])

    plt.scatter([],[],s=2,c='black',label=pointlabel)

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
        plt.savefig(save_path, bbox_inches='tight', pad_inches=.03)
    plt.show()

# def plot_everything(res, seperate=False):
#     rot_by_label = res[0]
#     angle_by_label = res[1]
#     scale_by_label = res[3]
#     shear_by_label = res[5]
#     if seperate:
#         for i in range(10):
#             print('Plotting label', i)
#             plot_angles(rot_by_label[i], angle_by_label[i])
#             plot_angles(rot_by_label[i], 100 * shear_by_label[i])
#             plot_angles(rot_by_label[i], 50 * scale_by_label[i][:,0])
#             plot_angles(rot_by_label[i], 50 * scale_by_label[i][:,1])

#             plt.hist(shear_by_label[i], 50, range=(-1.25, 1.25))
#             plt.figure()
#             plt.hist(scale_by_label[i][:,0], 50, range=(0.5, 2.5))
#             plt.figure()
#             plt.hist(scale_by_label[i][:,1], 50, range=(0.5, 2.5))
#             plt.show()
#     else:
#         rotated_angles = np.concatenate(rot_by_label)
#         plot_angles(rotated_angles, np.concatenate(angle_by_label), save_path=save_path, title=title)
#         plot_angles(rotated_angles, 100 * np.concatenate(shear_by_label))
#         plot_angles(rotated_angles, 50 * np.concatenate(scale_by_label)[:,0])
#         plot_angles(rotated_angles, 50 * np.concatenate(scale_by_label)[:,1])

def plot_distance(tran=None, pred=None, res=None, save_path='', title=''):
    if res is not None:
        assert tran is None and pred is None
        tran = np.concatenate(res[0])
        pred = np.concatenate(res[1])
    assert title == '', "Not implemented yet"
    heatmap, xedges, yedges = np.histogram2d(
        tran[:,0], pred[:,0], bins=32, range=[[-16,16],[-16,16]])
    extent = [-16, 16, -16, 16]
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.title('x')

    plt.figure()

    heatmap, xedges, yedges = np.histogram2d(
        tran[:,1], pred[:,1], bins=32, range=[[-16,16],[-16,16]])
    extent = [-16, 16, -16, 16]
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.title('y')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=.03)
    plt.show()

def plot_scale(logscale=None, pred=None, res=None, save_path='', title=''):
    if res is not None:
        assert logscale is None and pred is None
        logscale = np.concatenate(res[0])
        pred = np.concatenate(res[1])
    # blues, reds = get_cmap('Blues'), get_cmap('Reds')
    # blues._init()
    # blues._lut[:,-1] = np.linspace(0, 0.5, blues.N+3)
    # reds._init()
    # reds._lut[:,-1] = np.linspace(0, 0.5, reds.N+3)

    heatmap1, xedges, yedges = np.histogram2d(
        logscale, np.log2(pred[:,0]), bins=50, range=[[-1.5,2.5],[-1.5,2.5]])
    # extent = [-1.5,2.5,-1.5,2.5]
    # plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=blues)
    # plt.title('x')
    # plt.figure()

    heatmap2, xedges, yedges = np.histogram2d(
        logscale, np.log2(pred[:,1]), bins=50, range=[[-1.5,2.5],[-1.5,2.5]])
    extent = [-1.5,2.5,-1.5,2.5]
    # plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=reds)

    rgb = np.stack([heatmap1.T, heatmap2.T, np.mean([heatmap1.T, heatmap2.T],axis=0)], axis=-1)
    rgb = (np.ones_like(rgb) - rgb / np.max(rgb))
    plt.imshow(rgb, extent=extent, origin='lower')

    plt.xticks([-1, 0, 1, 2], ['0.5', '1', '2', '4'])
    plt.yticks([-1, 0, 1, 2], ['0.5', '1', '2', '4'])
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=.03)
    plt.show()


def transformation_statistics(model=0, plot=True, di=None, transform='rotate',
                              normalize=None, epochs=1, save_path='', title=''):
    global untransformed_test

    assert transform in ['rotate', 'translate', 'scale']

    if type(model) == int:
        model = get_model(model, di=di)
        _, untransformed_test = data.mnist(rotate=False, normalize=False, translate=False)
    elif 'untransformed_test' not in globals():
        _, untransformed_test = data.mnist(rotate=False, normalize=False, translate=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    transformation = np.zeros((0,2) if transform == 'translate' else (0,1))
    tran_by_label = []

    angles = np.array([])
    distances = np.zeros((0,2))
    scales = np.zeros((0,2))
    dets = np.array([])
    shears = np.array([])
    angle_by_label = []
    distance_by_label = []
    scale_by_label = []
    det_by_label = []
    shear_by_label = []

    labels = np.array([])

    with torch.no_grad():
        model.eval()
        if transform == 'translate':
            noise = data.MNIST_noise(60)
        elif transform == 'scale':
            noise = data.MNIST_noise(112, scale=True)
        for epoch in range(epochs):
            for x, y in untransformed_test:
                if transform == 'rotate':
                    angle = np.random.uniform(-90, 90, x.shape[0])
                    transformation = np.append(transformation, angle)
                    transformed = torch.tensor([
                        rotate(im[0], a) for im, a in zip(x, angle)
                    ], dtype=torch.float).reshape(-1, 1, 28, 28)
                    if normalize or normalize is None:
                        transformed = (transformed - 0.1307) / 0.3081

                elif transform == 'translate':
                    distance = np.random.randint(-16, 17, (x.shape[0], 2))
                    transformation = np.append(transformation, distance, axis=0)
                    transformed = torch.zeros(x.shape[0], 1, 60, 60, dtype=torch.float)
                    for i,(im,(xd,yd)) in enumerate(zip(x, distance)):
                        transformed[i, 0, 16-yd : 44-yd, 16+xd : 44+xd] = im[0]
                        transformed[i] = noise(transformed[i])
                    if normalize is True or (normalize is None and d['normalize']):
                        transformed = (transformed - 0.0363) / 0.1870

                elif transform == 'scale':
                    logscale = np.random.uniform(-1, 2, x.shape[0])
                    transformation = np.append(transformation, logscale)
                    scale = np.power(2, logscale)
                    transformed = F.pad(x, (42,42,42,42))
                    for i,s in enumerate(scale):
                        transformed[i] = tvF.to_tensor(tvF.affine(
                            tvF.to_pil_image(transformed[i]),
                            angle=0, translate=(0,0), shear=0, scale = s,
                            resample = PIL.Image.BILINEAR, fillcolor = 0))
                        transformed[i] = noise(transformed[i])
                    if normalize is True or (normalize is None and d['normalize']):
                        transformed = (transformed - 0.0414) / 0.1751

                theta = model.localization[0](model.pre_stn[0](transformed.to(device))).cpu()
                angle, shear, sx, sy, det = angle_from_matrix(theta, all_transformations=True)
                scale = np.stack((sx, sy), axis=1)
                distance = distance_from_matrix(theta,
                    mp = len(model.pre_stn[0]) > 0 and not d['loop'])

                angles = np.append(angles, angle)
                distances = np.append(distances, distance, axis=0)
                scales = np.append(scales, scale, axis=0)
                dets = np.append(dets, det)
                shears = np.append(shears, shear)

                labels = np.append(labels, y)

    variance = 0
    for i in range(10):
        indices = labels==i
        tran_by_label.append(transformation[indices])
        angle_by_label.append(angles[indices])
        distance_by_label.append(distances[indices])
        scale_by_label.append(scales[indices])
        det_by_label.append(dets[indices])
        shear_by_label.append(shears[indices])
        
        transformations = tran_by_label[i]
        if transform == 'rotate':
            predictions = angle_by_label[i]
        elif transform == 'translate':
            predictions = distance_by_label[i]
        elif transform == 'scale':
            # predictions = np.log2(scale_by_label[i])
            transformations = 2*transformations
            predictions = np.log2(det_by_label[i])
        s = (sum(transformations) + sum(predictions)) / len(transformations)
        variance += sum([np.linalg.norm(t+p - s)**2 for t,p in zip(transformations,predictions)])


    print('Standard deviation', np.sqrt(variance / (epochs * 10000)))

    if plot:
        if transform == 'rotate':
            plot_angles(rot=transformation, pred=angles, save_path=save_path, title=title)
        elif transform == 'translate':
            plot_distance(tran=transformation, pred=distances, save_path=save_path, title=title)
        else:
            plot_scale(logscale=transformation, pred=scales, save_path=save_path, title=title)

    return tran_by_label, angle_by_label, distance_by_label, scale_by_label, det_by_label, shear_by_label


def average_n(res, n):
    for run in res:
        s = 0
        for label in run[n]:
            s += sum(label)
        s /= len(untransformed_test.dataset)
        print(s)


def min_angle_dist(angles):
    sums = np.zeros_like(angles)
    for i,a1 in enumerate(angles):
        for a2 in angles:
            if a1 != a2:
                diff = (a1 - a2) % 180
                sums[i] += min(diff, 180-diff)
    return min(sums)

def plankton_rotation_statistics(model=0, di=None, samples=5, printprogress=True):
    global untransformed_test

    if type(model) == int:
        model = get_model(model, di=di)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # transformation = np.zeros((0,samples))
    # tran_by_label = []

    angles = np.zeros((0,samples))
    # distances = np.zeros((0,2,samples))
    # scales = np.zeros((0,2,samples))
    # dets = np.array([])
    # shears = np.array([])
    # angle_by_label = []
    # distance_by_label = []
    # scale_by_label = []
    # det_by_label = []
    # shear_by_label = []

    labels = np.array([])

    angle = np.linspace(-180, 180, num=samples, endpoint=False)

    l = len(untransformed_test.dataset)
    with torch.no_grad():
        model.eval()
        for i,(x,y) in enumerate(untransformed_test):
            if printprogress and i % 5 == 0:
                print(i*128 / l)
            # transformation = np.append(transformation, angle, axis=0)

            temp_angles = np.zeros((x.shape[0], samples))
            for s in range(samples):
                transformed = torch.tensor([
                    rotate(im[0], angle[s]) for im in x
                ], dtype=torch.float).reshape(-1, 1, 95, 95)

                # theta = model.localization[0](model.pre_stn[0](transformed.to(device))).cpu()
                if not d['loop']:
                    assert not d.get('batchnorm')
                    x2 = transformed.to(device)
                    base_theta = torch.eye(3).to(device)
                    for i,m in enumerate(model.pre_stn):
                        if i != 0:
                            x2 = model.stn(theta[:,0:2,:], loc_input)
                        if i == 0 or len(m) > 0:
                            loc_input = m(x2)
                            loc_output = model.localization[i](loc_input)
                            theta = base_theta
                        else:
                            loc_output = model.localization[i](x2)
                        mat = F.pad(loc_output, (0,3)).view((-1,3,3))
                        mat[:,2,2] = 1
                        theta = torch.matmul(theta,mat)
                else:
                    assert not d.get('batchnorm')
                    transformed = transformed.to(device)
                    theta = torch.eye(3).to(device)
                    x2 = transformed
                    for i,m in enumerate(model.loop_models):
                        if i != 0:
                            x2 = model.stn(theta[:,0:2,:], transformed)
                        localization_output = model.localization[i](m(x2))
                        mat = F.pad(localization_output, (0,3)).view((-1,3,3))
                        mat[:,2,2] = 1
                        theta = torch.matmul(theta,mat)
                theta = theta[:,0:2,:].view(-1,6).cpu()

                temp_angles[:,s] = angle_from_matrix(theta, all_transformations=False)
                # angle, shear, sx, sy, det = angle_from_matrix(theta, all_transformations=True)
                # scale = np.stack((sx, sy), axis=1)
                # distance = distance_from_matrix(theta)

            angles = np.append(angles, temp_angles, axis=0)

            # angles = np.append(angles, angle)
            # distances = np.append(distances, distance, axis=0)
            # scales = np.append(scales, scale, axis=0)
            # dets = np.append(dets, det)
            # shears = np.append(shears, shear)

            # labels = np.append(labels, y)

    tot_trans = angle+angles
    total_angle_dist = sum(min_angle_dist(a) for a in tot_trans)
    mean_angle_dist = total_angle_dist / (samples - 1) / l
    print(mean_angle_dist)

    return (mean_angle_dist)

    # variance = 0
    # for i in range(10):
    #     indices = labels==i
    #     tran_by_label.append(transformation[indices])
    #     # angle_by_label.append(angles[indices])
    #     # distance_by_label.append(distances[indices])
    #     # scale_by_label.append(scales[indices])
    #     # det_by_label.append(dets[indices])
    #     # shear_by_label.append(shears[indices])
        
    #     transformations = tran_by_label[i]
    #     predictions = angle_by_label[i]
    #     s = (sum(transformations) + sum(predictions)) / len(transformations)
    #     variance += sum([np.linalg.norm(t+p - s)**2 for t,p in zip(transformations,predictions)])



### GRADIENT STATISTICS ###

def hook(x):
    print('Shape', x.shape)
    print('Median abs', torch.median(torch.abs(x)))
    print('Max', torch.max(x), 'Min', torch.min(x))
    summed = torch.sum(x, dim=0)
    print('Median sum', torch.median(torch.abs(summed)))
    print('Max', torch.max(summed), 'Min', torch.min(summed))
    print('Vector length', torch.norm(summed.view(-1)))

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
            theta = torch.eye(3)
            for i,m in enumerate(model.loop_models):
                loc_input = m(x)
                # if x.requires_grad:
                #     loc_input.register_hook(lambda a: print('loc input', hook(a), '\n'))
                # model.localization[i].register_backward_hook(module_hook)
                loc_output = model.localization[i](loc_input)
                # loc_output.register_hook(lambda a: print('loc output', hook(a), '\n'))
                mat = F.pad(loc_output, (0,3)).view((-1,3,3))
                mat[:,2,2] = 1
                theta = torch.matmul(theta,mat)
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
                    print(torch.norm(p.grad.view(-1)))
                    print()

                print('Localization', i)
                for p in l.parameters():
                    print(p.grad.shape)
                    print(torch.norm(p.grad.view(-1)))
                    print()

            print('Final layers')
            for p in model.final_layers.parameters():
                print(p.grad.shape)
                print(torch.norm(p.grad.view(-1)))
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
                print(torch.norm(p.grad.view(-1)))
                print()

            print('Localization', i)
            for p in l.parameters():
                print(p.grad.shape)
                print(torch.norm(p.grad.view(-1)))
                print()

        print('Final layers')
        for p in model.final_layers.parameters():
            print(p.grad.shape)
            print(torch.norm(p.grad.view(-1)))
            print()


### VISUALISE WEIGHTS ###

def plot_first_layer(model1=0, model2=0, di1=None, di2=None):
    if type(model1) == int:
        model1 = get_model(model1, di=di1)
    if type(model2) == int:
        model2 = get_model(model2, di=di2)

    layer1 = model1.pre_stn[0] if len(model1.pre_stn[0]) else model1.final_layers[0]
    while type(layer1) == torch.nn.Sequential:
        layer1 = layer1[0]
    print(layer1)

    layer2 = model2.pre_stn[0] if len(model2.pre_stn[0]) else model2.final_layers[0]
    print('1',layer2)
    while type(layer2) == torch.nn.Sequential:
        layer2 = layer2[0]
        print('2',layer2)
    print('3',layer2)

    filters1 = next(layer1.parameters())
    filters2 = next(layer2.parameters())

    mi = min(torch.min(filters1), torch.min(filters2))
    ma = max(torch.max(filters1), torch.max(filters2))
    print('mi',mi)
    print('ma',ma)

    fig, axs = plt.subplots(8, 8, sharex='col', sharey='row', figsize=(11,9),
                            gridspec_kw={'hspace': 0.02, 'wspace': 0.02})
    for i,f in enumerate(filters1):
        axs[i%8, i//8].imshow(f[0].detach().numpy(), vmin=mi, vmax=ma)
    for i,f in enumerate(filters2):
        pcm = axs[i%8, 4+i//8].imshow(f[0].detach().numpy(), vmin=mi, vmax=ma)

    fig.colorbar(pcm, ax=axs)

    plt.show()
