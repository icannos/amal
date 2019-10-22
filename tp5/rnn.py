import torch
import torch.nn as nn


class RnnLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.output_dim = output_dim
        self.input_dim = input_dim

        self.inputLinear = nn.Linear(input_dim, output_dim, bias=False)
        self.hiddenLinear = nn.Linear(output_dim, output_dim, bias=True)

        self.activation = nn.Sigmoid()

    def forward(self, X):
        hidden = [torch.zeros((X.shape[1], self.output_dim), requires_grad=False).double()]
        for i in range(X.shape[0]):
            hidden.append(self.activation(self.one_step(X[i], hidden[i])))

        return hidden[-1]

    def one_step(self, x, h):
        ht = self.hiddenLinear(h)
        input_t = self.inputLinear(x)

        return input_t + ht



