from sentiment_dataset import load_dataset
from torch.utils.data import Dataset, DataLoader
import sa_models as sa
import torch.nn as nn
import torch
import torch.utils.data as tud
from pathlib import Path
from tp5_preprocess import TextDataset
import time

savepath = Path("checkpoints/save.chkpt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset()

print(dataset.text)
dataset_train = DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=TextDataset.collate)

epochs = 100

if savepath.is_file():
    try:
        with savepath.open("rb") as fp:
            model = torch.load(fp)
    except Exception:
        module = sa.SaModel1()
        model = sa.Model(model=module, LossFunction=nn.functional.cross_entropy)
else:
    module = sa.SaModel1()
    model = sa.Model(model=module, LossFunction=nn.functional.cross_entropy, cuda=False)

for e in range(model.curr_epoch, epochs):
    print(e)
    acc = torch.Tensor([0])
    n = 0
    for x, y in dataset_train:
        model.training_step(x, y)
        acc = model.accuracy(x,y)

        n += 1

        print(acc)

    with savepath.open("wb") as fp:
        model.curr_epoch = e
        torch.save(model, fp)
