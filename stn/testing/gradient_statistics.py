import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy

import eval

train_loader = None # just to get rid of errors
directory = 'svhn/tuning/hook/lr003d10s120k_svhns0123loop_llr005'
eval.load_data(directory)

version = 'final'
model = 0

models = [eval.get_model(model, version=version, llr=llr) for llr in [False, 0]]
optims = [t.optim.SGD(model.parameters(), lr=1) for model in models]
grads = [[],[]]

def get_grads(runs):
    for i,(x,y) in enumerate(eval.train_loader):
        print(i)
        for grad,optim,model in zip(grads,optims,models):
            model.train()
            optim.zero_grad()
            output = model(x)
            loss = sum(F.cross_entropy(output[i],y[:,i]) for i in range(5))
            loss.backward()

            pre_stn_grad = []
            loc_grad = []
            for (l,m) in(zip(model.localization, model.pre_stn)):
                pre_stn_grad.append([p.grad.clone() for p in m.parameters()])
                loc_grad.append([p.grad.clone() for p in l.parameters()])

            final_grad = [p.grad.clone() for p in model.final_layers.parameters()]

            grad.append([pre_stn_grad, loc_grad, final_grad])

        if i == runs-1:
            break

def sum_tensors(grads):
    prestngrads0 = [[sum(grads[0][run][0][l][p] for run in range(len(grads[0])))
      for p in range(len(grads[0][ 0 ][0][l]))]
      for l in range(len(grads[0][ 0 ][0]))]
    prestngrads1 = [[sum(grads[1][run][0][l][p] for run in range(len(grads[0])))
      for p in range(len(grads[1][ 0 ][0][l]))]
      for l in range(len(grads[1][ 0 ][0]))]

    loc_grads0   = [[sum(grads[0][run][1][l][p] for run in range(len(grads[0])))
      for p in range(len(grads[0][ 0 ][1][l]))]
      for l in range(len(grads[0][ 0 ][1]))]
    loc_grads1   = [[sum(grads[1][run][1][l][p] for run in range(len(grads[0])))
      for p in range(len(grads[1][ 0 ][1][l]))]
      for l in range(len(grads[1][ 0 ][1]))]

    final_grads0 = [sum(grads[0][run][2][p] for run in range(len(grads[0]))) for p in range(len(grads[0][0][2]))]
    final_grads1 = [sum(grads[1][run][2][p] for run in range(len(grads[0]))) for p in range(len(grads[0][0][2]))]

    return prestngrads0, prestngrads1, loc_grads0, loc_grads1, final_grads0, final_grads1

def sum_norms(grads):
    prestngrads0 = [[sum(t.norm(grads[0][run][0][l][p]) for run in range(len(grads[0])))
            for p in range(len(grads[0][ 0 ][0][l]))]
            for l in range(len(grads[0][ 0 ][0]))]
    prestngrads1 = [[sum(t.norm(grads[1][run][0][l][p]) for run in range(len(grads[0])))
            for p in range(len(grads[1][ 0 ][0][l]))]
            for l in range(len(grads[1][ 0 ][0]))]
    loc_grads0   = [[sum(t.norm(grads[0][run][1][l][p]) for run in range(len(grads[0])))
            for p in range(len(grads[0][ 0 ][1][l]))]
            for l in range(len(grads[0][ 0 ][1]))]
    loc_grads1   = [[sum(t.norm(grads[1][run][1][l][p]) for run in range(len(grads[0])))
            for p in range(len(grads[1][ 0 ][1][l]))]
            for l in range(len(grads[1][ 0 ][1]))]
    final_grads0 = [sum(t.norm(grads[0][run][2][p]) for run in range(len(grads[0]))) for p in range(len(grads[0][0][2]))]
    final_grads1 = [sum(t.norm(grads[1][run][2][p]) for run in range(len(grads[0]))) for p in range(len(grads[0][0][2]))]
    return prestngrads0, prestngrads1, loc_grads0, loc_grads1, final_grads0, final_grads1