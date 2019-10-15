import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from rnn import RnnLayer
import torch
from torch import nn
import numpy as np
import string
from torch.utils.data import Dataset, DataLoader

embedding_size = 100

def decode_text(encoded_text, idx2char):
    return "".join([idx2char[idx] for idx in encoded_text])

def encode_text(text, char2idx):
    return [char2idx[c] for c in text]


class TempPred(Dataset):
    def __init__(self, seqlength=16, n_seqs=10000, train=True) -> None:
        super().__init__()

        text = open("trump_full_speech.txt", 'r').read()

        idx2char = {0:''}
        char2idx = {'':0}

        idx = 1
        for c in text:
            if c not in char2idx:
                char2idx[c] = idx
                idx2char[idx] = c
                idx+=1

        if train:
            encoded_text = np.array([char2idx[c] for c in text])[:150000]
        else:
            encoded_text = np.array([char2idx[c] for c in text])[150000:]

        self.idx2char = idx2char
        self.char2idx = char2idx

        X = []
        Y = []
        n = encoded_text.shape[0]

        for i in range(n_seqs):
            id = np.random.randint(n-seqlength-1)
            X.append(encoded_text[id:id+seqlength])
            Y.append(encoded_text[id+seqlength])

        self.train_x = np.array(X)
        self.train_y = np.array(Y)

    def __getitem__(self, item):
        return self.train_x[item], self.train_y[item]

    def __len__(self):
        return self.train_x.shape[0]

def generate(model, seed, lenght=400):
    seq = list(np.copy(seed))
    for i in range(lenght):
        x = np.array(seq[-16:])
        x = torch.tensor(x)
        pc = nn.functional.softmax(model(x))[0]
        c = np.random.choice(pc.detach().numpy().shape[0], p=pc.detach().numpy())
        seq.append(c)

    return seq

dataset = TempPred()
data = DataLoader(dataset, shuffle=True, batch_size=512)
datatest = DataLoader(TempPred(train=False), shuffle=True, batch_size=512)

embedding = nn.Embedding(len(dataset.idx2char), embedding_size)
rnnlayer = RnnLayer(embedding_size, 64)
classifier = nn.Linear(64, len(dataset.idx2char), bias=True)

model = nn.Sequential(embedding, rnnlayer, classifier)

loss = nn.CrossEntropyLoss()

optimizer = Adam(params=list(model.parameters()))

epoch = 1000
writer = SummaryWriter()

model = model.double()

c = 0
for step in range(1000):
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

        writer.add_scalar("CharPred/CrossEntropy", l, c)
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

    if (step+1) % 15 == 1:
        seed = encode_text("I will make America great again", char2idx=dataset.char2idx)
        text = generate(model, seed=seed)
        print(decode_text(text, dataset.idx2char))


# Exemple après 50 itérations
# I will make America great againd of ion.
# and yon so cand engas thanavues Hrour Pmed
# an ame, sor coud gmide Wint japerustach thatas te
# that or and at doupeat., and trouneve, So aug bmend ouden
# fo0 hor pnas on to bel Weey Wi deatanguse to eod ionk prrite
# eveaclinte. Aray,. dointanky iwr mur hampyitingall Iappremiowl'mpadust
# wianastm roustowr bem0: inderilatey s pust Avighergter und aco tow too re
# Thatiof tot Dom a I ud ik, yor, I

# Exemple après 300 itérations
# I will make America great again replan and of jower, the Tromica,
# be sone sertiamanetion." Whink. when thank peoth shonatirgeg
# the villables and hen the dowed ming noing. We eepalyon liblubiom.
# A ow, bring many lompero yount ould billiked billinedingen forhes.
# It and what it Tricn. It Ou'r mute. I thank. Tridntt Miked a tthat prombeit nowh
# and ther morn. OK of these mussind or
# work--Clight that well that weold fhenders iwa lige