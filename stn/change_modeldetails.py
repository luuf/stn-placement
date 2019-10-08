#%%
import torch as t

def change_modeldetails(data_dir, normalize=None, batchnorm=None):
    directory = '../experiments/'+data_dir

    d = t.load(directory+"/model_details")
    print('d1', d)
    if normalize is not None:
        d['normalize'] = normalize
    if batchnorm is not None:
        d['batchnorm'] = batchnorm

    print('d2', d)

    t.save(
        d, # why is it not saved ??
        directory + '/model_details',
    )
