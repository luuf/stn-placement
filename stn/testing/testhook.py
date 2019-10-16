import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Test_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2,2)
        self.l2 = nn.Linear(2,2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x.register_hook(lambda v: 10*v)
        x = self.l2(x)
        return x

m = Test_model()

o = t.optim.SGD(
    params=[{'params':m.l1.parameters(), 'lr':0.1},
            {'params':m.l2.parameters(), 'lr':0.1}],
)

b = [p.detach() for p in m.parameters()]
print('Before', b)

ce = nn.CrossEntropyLoss()

for i in range(10):
    m.train()
    o.zero_grad()
    output = m(t.tensor([[1,1]], dtype=t.float))
    print(output)
    loss = ce(output, t.tensor([1], dtype=t.long))

    loss.backward()
    o.step()

a = [p.detach() for p in m.parameters()]
print('After', i, a)

# print('difference', [p2-p1 for (p1,p2) in zip(a,b)])