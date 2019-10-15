
# webia.lip6.fr/~bpiwowari/amal.requirement.txt

from torch.utils.data import Dataset, DataLoader
from datamaestro import prepare_dataset
from datamaestro import prepare_dataset
import numpy as np
from torch import nn
from torch.optim import sgd
from torch.utils.tensorboard import SummaryWriter
import torch

learning_rate = 0.001

device = torch.device("cuda")

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

    def __getitem__(self, item):
        return self.train_x[item]

    def __len__(self):
        return self.train_x.shape[0]


class Autoencoder(nn.Module):
    def __init__(self, encoded_dim=10):
        super(Autoencoder, self).__init__()

        self.W = torch.nn.Parameter(torch.zeros(28*28, encoded_dim), requires_grad=True)
        self.b1 = torch.nn.Parameter(torch.zeros(encoded_dim), requires_grad=True)
        self.b2 = torch.nn.Parameter(torch.zeros(28*28), requires_grad=True)


    def encoder(self, x):
        return x.matmul(self.W) + self.b1

    def decoder(self, x):
        return x.matmul(self.W.t()) + self.b2

    def forward(self, x):
        return self.decoder(self.encoder(x))


data = DataLoader(myMNIST(), shuffle=True, batch_size=256)


auc = Autoencoder()
optimizer =sgd.SGD(list(auc.parameters()), lr=learning_rate)

writer = SummaryWriter()

auc =auc.to(device)

for step in range(100):
    print(step)
    for x in data:
        x=x.type(torch.FloatTensor)
        x = x.to(device)
        xchap = auc(x)

        l = nn.MSELoss()(xchap, x)

        l.backward()

        optimizer.step()

        auc.zero_grad()
        optimizer.zero_grad()

        writer.add_scalar("Autoencoder/MSE", l, step)







