#%%
import torch as t
import modelstorch as models
import datatorch as data
import matplotlib.pyplot as plt
import numpy as np

directory = "../experiments/dropout/STCNN/"

d = t.load(directory+"model_details")
trainloader, testloader = data.data_dic[d['dataset']](d['rotate'])
#%% Functions
def get_model(prefix):
    model = models.Net(
        models.model_dic[d['model']](d['model_parameters']),
        models.localization_dic[d['localization']](d['localization_parameters']),
        d['stn_placement'],
        d['loop'],
        trainloader.dataset[0][0].shape
    )
    model.load_state_dict(t.load(
        directory+str(prefix)+"final",
        map_location='cpu'
    ))
    # model.load_state_dict(t.load(directory+prefix+"ckpt"+"100"))
    return model

def print_history(prefixes=[0,1,2]):
    if type(prefixes) == int:
        prefixes = [prefixes]
    for prefix in prefixes:
        history = t.load(directory + str(prefix) + "history")
        print('PREFIX', prefix)
        print('Max test_acc', np.argmax(history['test_acc']))
        print('Max train_acc', np.argmax(history['train_acc']))
        print('Final test_acc', history['test_acc'][-1])
        print('Final train_acc', history['train_acc'][-1])
        plt.plot(history['train_acc'][:])
        plt.plot(history['test_acc'][:])
        # plt.plot(history['train_loss'][:])
        # plt.plot(history['test_loss'][:])
        plt.show()

def test_stn(model, n=1):
    if type(model) == int:
        model = get_model(model)
    batch = next(iter(trainloader))[0][:n]
    for image,transformed in zip(batch, model.stn(model.pre_stn(batch), batch)):
        plt.subplot(1,2,1)
        plt.imshow(np.moveaxis(np.array(image),0,-1))
        plt.subplot(1,2,2)
        plt.imshow(np.moveaxis(transformed.detach().numpy(),0,-1))
        plt.show()

#%%
