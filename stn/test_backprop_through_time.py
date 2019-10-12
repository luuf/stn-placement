#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(2,2)

    def hook(self,x):
        print(x)

    def forward(self, x):
        x = F.relu(self.l(x))
        x.register_hook(self.hook)
        return x

class StopGradient(nn.Module):
    def hook(self, x):
        return 0*x

    def forward(self, x):
        x.register_hook(self.hook)
        return x


        
net = Net()
stop = StopGradient()
net.train()

o = t.optim.SGD(net.parameters(), lr=0.1)
x = torch.tensor([[1.,1.]])

o.zero_grad()
x = net(x)
x = net(x)
x = stop(x)
x = net(x)

ce = nn.CrossEntropyLoss()
print(x.shape)
loss = ce(x, torch.tensor([0], dtype=torch.long))
loss.backward()
o.step()