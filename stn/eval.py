#%%
import torch as t
import matplotlib.pyplot as plt
import numpy as np
import models
import data

directory = "../experiments/svhn/extra/STCNN0/"

d = t.load(directory+"model_details")
if d['dataset'] in data.data_dict:
    train_loader, test_loader = data.data_dict[d['dataset']](
        d['rotate'], normalize=False)
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
        directory+str(prefix)+"final",
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

def test_stn(model, n=1):
    if type(model) == int:
        model = get_model(model)
    model.eval()
    batch = next(iter(test_loader))[0][:n]
    theta = model.localization[0](model.pre_stn[0](batch))
    for image,transformed in zip(batch, model.stn(theta.view(-1,2,3), batch)):
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

#%%
