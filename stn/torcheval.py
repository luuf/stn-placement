#%%
import torch as t
import modelstorch as models
import matplotlib.pyplot as plt
import numpy as np

directory = "../torch_experiments/CNNfalse/"
prefix = "0"

d = t.load(directory+"model_details")
model = models.Net(
    models.model_dic[d['model']](d['model_parameters']),
    models.localization_dic[d['localization']](d['localization_parameters']),
    d['stn_placement'],
    d['loop']
)
model.load_state_dict(t.load(
    directory+prefix+"final",
    map_location='cpu'
))
# model.load_state_dict(t.load(directory+prefix+"ckpt"+"100"))

history = t.load(directory + prefix + "history")
# plt.plot(history['train_acc'][300:])
# plt.plot(history['test_acc'][300:])
plt.plot(history['train_loss'][400:])
plt.plot(history['test_loss'][400:])
plt.show()
print('Max test_acc', np.argmax(history['test_acc']))
print('Max train_acc', np.argmax(history['train_acc']))
print('Final test_acc', history['test_acc'][-1])
print('Final train_acc', history['train_acc'][-1])

#%%
