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

        return y_grad, None


## Pour utiliser la fonction

## Pour telecharger le dataset Boston
ds=prepare_dataset("edu.uci.boston")
fields, data =ds.files.data()

n = data.shape[0]

regling = linear1()
mse = MSE()


learning_rate = 0.1

w = torch.rand(13, requires_grad=True, dtype=torch.double)
b = torch.rand(1, requires_grad=True, dtype=torch.double)

writer = SummaryWriter()

for step in range(100):
    idx = np.random.randint(0, n)

    for i in range(data.shape[0]):
        acc_gradw = None
        acc_gradb = None

        x = torch.DoubleTensor(data[i, :-1]/100)
        y = torch.DoubleTensor([data[i, -1] / 100])

        ctx_lin = Context()
        ycha = linear1.forward(ctx_lin, x, w, b)

        ctx_mse = Context()
        l = mse.forward(ctx_mse, ycha, y)

        grad_output = mse.backward(ctx_mse, 1)

        gradx, gradw, gradb = linear1.backward(ctx_lin, grad_output[0])

        if acc_gradw is not None:
            acc_gradw += gradw
        else:
            acc_gradw = gradw

        if acc_gradb is not None:
            acc_gradb += gradb
        else:
            acc_gradb = gradb

    w = w - learning_rate * (gradw / data.shape[0])
    b = b - learning_rate * (gradb /data.shape[0])

    writer.add_scalar("Batch/Loss/MSE", l, step)




