import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.optim import sgd
from torch.utils.tensorboard import SummaryWriter
from rnn import RnnLayer
import torch
from torch import nn
import numpy as np

from torch.utils.data import Dataset, DataLoader


df = pd.read_csv("tempAMAL_train.csv")


class TempPred(Dataset):
    def __init__(self, seqlength=4, n_per_city=5000, train=True) -> None:
        super().__init__()

        if train:
            df = pd.read_csv("tempAMAL_train.csv")[:2000]
        else:
            df = pd.read_csv("tempAMAL_train.csv")[2000:]

        dfloc = pd.read_csv("city_attributes.csv")

        df = df.dropna()

        classes_name = list(df.columns)[1:][0:5]

        X = []
        Y = []

        for c_id, c in enumerate(classes_name):
            for i in range(n_per_city):
                d = df[c].values
                idx = np.random.randint(d.shape[0] - seqlength - 1)
                X.append(d[idx:idx + seqlength])
                Y.append(c_id)
        x = np.array(X)
        self.train_x = np.reshape(x, [*x.shape, 1]) / 300
        self.train_y = np.array(Y)

    def __getitem__(self, item):
        return self.train_x[item], self.train_y[item]

    def __len__(self):
        return self.train_x.shape[0]


rnnlayer = RnnLayer(1, 16)
classifier = nn.Linear(16, 5, bias=True)

model = nn.Sequential(rnnlayer, classifier)

loss = nn.CrossEntropyLoss()

optimizer = sgd.SGD(params=list(model.parameters()), lr=0.001)

epoch = 1000
writer = SummaryWriter()

model = model.double()

data = DataLoader(TempPred(), shuffle=True, batch_size=512)
datatest = DataLoader(TempPred(train=False), shuffle=True, batch_size=512)

c = 0
for step in range(100):
    print(step)
    for x,y in data:
        optimizer.zero_grad()
        model.zero_grad()
        x = x.transpose(1,0)

        ychap = model(x)
        l = loss(ychap, y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar("TempPred/MSE", l, c)
        c+=1

    n = 0
    totloss = 0
    for x,y in datatest:
        n+=1
        x = x.transpose(1, 0)
        totloss += loss(model(x), y)
    totloss /= n
    print("Training: ", l)
    print("Validation:", totloss)
