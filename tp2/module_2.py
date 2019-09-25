

import torch



# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck
from datamaestro import prepare_dataset
import numpy as np
import torch.nn as nn
from torch.optim import sgd

from torch.utils.tensorboard import SummaryWriter

learning_rate = 0.0001
minibatch_size = 256

# ============================================== #
ds=prepare_dataset("edu.uci.boston")
fields, data =ds.files.data()
n = data.shape[0]

writer = SummaryWriter()

f1 = nn.Linear(data.shape[1]-1, 10)
f2 = nn.Linear(10, 1)

# On utilise un container, il va aggréger les paramètres des différents modules
network = nn.Sequential(f1, nn.Tanh(), f2)

# Loss op
loss = nn.MSELoss()

# Notre optimizer travaillant sur les paramètres du modèle network
optimizer =sgd.SGD(list(network.parameters()), lr=learning_rate)

for step in range(1000):
    # On sample un minibatch
    idx = np.random.choice(data.shape[0], minibatch_size)

    x = torch.FloatTensor(data[idx, :-1] / 100)
    y = torch.FloatTensor([data[idx, -1] / 100])

    # On utilise le container pour générer la sortie
    output = network(x)

    l = loss(output, y)
    # On différencie
    l.backward()
    # un pas de descente de gradient
    optimizer.step()
    # Remise à 0 des gradients pour pas que ça explose
    network.zero_grad()
    optimizer.zero_grad()

    writer.add_scalar("Modules-Seq/sgd/Loss/MSE", l, step)




