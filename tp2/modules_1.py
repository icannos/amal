import torch
from torch.autograd import Function
from torch.autograd import gradcheck
from datamaestro import prepare_dataset
import numpy as np
import torch.nn as nn
from torch.optim import sgd

from torch.utils.tensorboard import SummaryWriter


## Pour telecharger le dataset Boston
ds=prepare_dataset("edu.uci.boston")
fields, data =ds.files.data()

n = data.shape[0]

learning_rate = 0.01
minibatch_size = 64


writer = SummaryWriter()

# On définit les différentes couches de notre modèle
f1 = nn.Linear(data.shape[1]-1, 10)
f2 = nn.Linear(10, 1)

loss = nn.MSELoss()

# On récupère tous les paramètres des différents modules
optimizer =sgd.SGD([*list(f1.parameters()), *list(f2.parameters())], lr=learning_rate)

for step in range(10000):
    idx = np.random.choice(data.shape[0], minibatch_size)

    x = torch.FloatTensor(data[idx, :-1] / 100)
    y = torch.FloatTensor([data[idx, -1] / 100])

    output = f2(nn.Tanh()(f1(x)))
    l = loss(output, y)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

    writer.add_scalar("Modules/sgd/Loss/MSE", l, step)




