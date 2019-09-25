# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck
from datamaestro import prepare_dataset
import numpy as np
from torch.optim import adadelta

from torch.utils.tensorboard import SummaryWriter

class Context:
    """Very simplified context object"""
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors

class linear1(Function):
    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w, b)
        prod = torch.dot(w, x)
        return prod + b


    @staticmethod
    def backward(ctx, grad_output):
        x,w, b = ctx.saved_tensors
        dfdx = grad_output * w
        dfdw = grad_output * x
        dfdb = grad_output

        return dfdx, dfdw, dfdb

class MSE(Function):
    @staticmethod
    def forward(ctx, y, target):
        ctx.save_for_backward(y, target)
        return torch.sum(torch.pow(y-target, 2))

    @staticmethod
    def backward(ctx, grad_outputs):
        y, target = ctx.saved_tensors
        y_grad = (2*y - 2*target) * grad_outputs

        return y_grad, None


learning_rate = 0.01

## Pour telecharger le dataset Boston
ds=prepare_dataset("edu.uci.boston")
fields, data =ds.files.data()

n = data.shape[0]

regling = linear1()
mse = MSE()


# parameters
w = torch.rand(13, requires_grad=True, dtype=torch.double)
b = torch.rand(1, requires_grad=True, dtype=torch.double)

writer = SummaryWriter()

# We define an optimizer and we give to it the paramets it should opptimize
optimizer = adadelta.Adadelta(params=(w,b))

for step in range(100):
    # On remet les gradients à 0 pour éviter de répercuter des gradients d'un élément à l'autre
    optimizer.zero_grad()

    l = None
    # J'ai eu des problèmes avec les opérations sur les batch donc j'accumule naive l'erreur sur un batch
    # Et je procède à la descente de gradient sur l'erreur aggrégée, c'est pas efficace mais ça fonctionne
    for i in range(data.shape[0]):

        x = torch.DoubleTensor(data[i, :-1]/100)
        y = torch.DoubleTensor([data[i, -1] / 100])

        y.requires_grad = False
        x.requires_grad = True

        ycha = linear1.apply(x, w, b)

        # on accumule l'erreur sur le batch
        if l is not None:
            l += mse.apply(ycha, y)
        else:
            l = mse.apply(ycha, y)

    l = l / data.shape[0]

    # Différentiation auto
    l.backward()
    optimizer.zero_grad()
    # pas de descente de de gr
    optimizer.step()

    writer.add_scalar("Batch-opti/Loss/MSE", l, step)




