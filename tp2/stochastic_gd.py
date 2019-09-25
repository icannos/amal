# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck
from datamaestro import prepare_dataset
import numpy as np

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

        return y_grad, y_grad


## Pour utiliser la fonction 

## Pour telecharger le dataset Boston
ds=prepare_dataset("edu.uci.boston")
fields, data =ds.files.data()

n = data.shape[0]

learning_rate = 0.001

# Parameters
w = torch.rand(13, requires_grad=True, dtype=torch.double)
b = torch.rand(1, requires_grad=True, dtype=torch.double)

writer = SummaryWriter()

for step in range(100):
    idx = np.random.randint(0, n)

    x = torch.DoubleTensor(data[idx, :-1]/100)
    y = torch.DoubleTensor([data[idx, -1] / 100])

    # Apparemment il me faut ça pour pas que ça explose
    y.requires_grad = False
    x.requires_grad = True

    # On calcule la sortie du modèle
    ycha = linear1.apply(x, w, b)

    l = MSE.apply(ycha, y)

    # Différentiation auto
    l.backward()

    with torch.no_grad():
        # ici on affecte aux variables w et b de nouvelles valeurs en utilisant data
        # utilise w = ... transformerait w en tensor alors que w.data =... permet d'affecter une valeur
        w.data = w - learning_rate * w.grad.data
        b.data = b - learning_rate * b.grad.data

        # On nettoie les gradients pour pas les conserver au tour suivant
        w.grad.data.zero_()
        b.grad.data.zero_()

    writer.add_scalar("Naive/Loss/MSE", l, step)




