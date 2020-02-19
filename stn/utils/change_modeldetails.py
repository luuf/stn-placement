#%%
import torch

def change_modeldetails(data_dir, normalize=None, batchnorm=None):
    directory = '../../experiments/'+data_dir

    d = torch.load(directory+"/model_details")
    print('d1', d)
    if normalize is not None:
        assert d.get('normalize') is None
        d['normalize'] = normalize
    if batchnorm is not None:
        assert d.get('batchnorm') is None
        d['batchnorm'] = batchnorm

    print('d2', d)

    torch.save(
        d, # why is it not saved ??
        directory + '/model_details',
    )
