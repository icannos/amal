import torch
from torch import nn
from torch.optim import Adam
import numpy as np


def causes(i, archi=((3, 1), (2, 2))):
    si, wi = archi[0]
    causes_idx = set(range(i*si, i*si+wi+1))

    return set.union(*[causes(j, archi[:-1]) for j in causes_idx])

class Model():
    def __init__(self, model, LossFunction, cuda=False):
        self.cuda = cuda
        self.LossFunction = LossFunction
        self.modele = model
        self.curr_epoch = 0
        self.optimizer = Adam(params=list(self.modele.parameters()))

        if self.cuda:
            self.modele = self.modele.to(device=cuda)

    def training_step(self, x ,y):
        if self.cuda:
            x,y = x.to(device=self.cuda), x.to(device=self.cuda)

        self.optimizer.zero_grad()

        l = self.loss(x, y)
        l.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

    def loss(self, x, y):
        if self.cuda:
            x,y = x.to(self.cuda), y.to(self.cuda)
        return self.LossFunction(self.modele(x), y)

    def accuracy(self, x, y):
        if self.cuda:
            x,y = x.to(self.cuda), y.to(self.cuda)
        return torch.mean(torch.eq(self.modele(x).argmax(axis=1),y).float())

    def __call__(self, x):
        if self.cuda:
            x = x.to(self.cuda)
        return self.modele(x)


class SaModel1(nn.Module):
    def __init__(self, vocab_size=1000, embedding_size=256):
        super().__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.conv1 = nn.Conv1d(self.embedding_size, 16, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=2, stride=1)

        self.classifier = nn.Linear(16, 3)

    def convnet(self, input):
        emb = self.embedding(input)
        emb = torch.transpose(emb, 1, 2)
        y = self.conv3(self.conv1(emb))
        val, argmax = torch.max(y, -1)

        return val, argmax

    def forward(self, input):
        val, _ = self.convnet(input)
        y = self.classifier(val)

        return y

