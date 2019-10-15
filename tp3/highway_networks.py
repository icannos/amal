import torch
from torch import nn


class HighwayLayer(nn.Module):
    def __init__(self, dim, basemodule, **kwargs):
        super().__init__()

        self.basemodule = basemodule
        self.dim = dim

        self.WT = torch.nn.Parameter(torch.zeros(self.dim, self.dim), requires_grad=True)
        self.bT = torch.nn.Parameter(torch.zeros(self.dim), requires_grad=True)

        self.H = basemodule(**kwargs)

    def T(self, x):
        return nn.Sigmoid()(x.matmul(self.Wt) + self.bT)

    def C(self, x):
        return 1-self.T(x)

    def forward(self, x):
        return self.H(x)*self.T(x) + x*self.C(x)


