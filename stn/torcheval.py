import torch as t
import modelstorch as models
import matplotlib.pyplot as plt

directory = "testEval/"
prefix = "1"

d = t.load(directory+"model_details")
model = models.Net(
    models.model_dic[d['model']](d['model_parameters']),
    models.localization_dic[d['localization']](d['localization_parameters']),
    d['stn_placement'],
    d['loop']
)
model.load_state_dict(t.load(directory+prefix+"final"))
# model.load_state_dict(t.load(directory+prefix+"ckpt"+"100"))

history = t.load(directory + prefix + "history")
plt.plot(history['train_acc'])
plt.plot(history['test_acc'])
plt.show()