import torch
from torch import nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from datamaestro import prepare_dataset
import numpy as np
from torch import nn
from torch.optim import sgd
from torch.utils.tensorboard import SummaryWriter
import torch

class myMNIST(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()

        self.train = train

        ds = prepare_dataset("com.lecun.mnist")

        if self.train:
            train_x, train_y = ds.files["train/images"].data(), ds.files["train/labels"].data()
        else:
            train_x, train_y = ds.files["test/images"].data(), ds.files["test/labels"].data()

        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]*train_x.shape[2])) / 255

        self.train_x = train_x
        self.train_y = train_y

    def __getitem__(self, item):
        return self.train_x[item], self.train_y[item]

    def __len__(self):
        return self.train_x.shape[0]

class HighwayLayer(nn.Module):
    def __init__(self, dim, basemodule, *args):
        super().__init__()

        self.basemodule = basemodule
        self.dim = dim

        self.HighWayPath = nn.Linear(dim, dim, bias=False)

        self.H = basemodule(*args)

    def T(self, x):
        return nn.Sigmoid()(self.HighWayPath(x))

    def C(self, x):
        return 1-self.T(x)

    def forward(self, x):
        return self.H(x)*self.T(x) + x*self.C(x)

class LinWActivation(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.lin = nn.Linear(*args)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.lin(x))




num_layers = 10

highway_layers = [HighwayLayer(64, nn.Linear, 64, 64)]
first_layer = LinWActivation(784, 64)

model = nn.Sequential(first_layer, *highway_layers, nn.Linear(64, 10))

loss = nn.CrossEntropyLoss()

data = DataLoader(myMNIST(), shuffle=True, batch_size=256)
optimizer =sgd.SGD(list(model.parameters()), lr=0.001)

writer = SummaryWriter()

model = model.float()

c = 0
for step in range(100):
    print(step)
    for i, (x, y) in enumerate(data):
        x=x.type(torch.FloatTensor)
        y = y.type(torch.LongTensor)
        ychap = model(x)

        l = loss(ychap, y)

        l.backward()

        optimizer.step()

        model.zero_grad()
        optimizer.zero_grad()

        writer.add_scalar("MNIST_classifier_highway/CrossEntropy", l, c)
        c+=1


