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

dataset = load_dataset()

print(dataset.text)
dataset_train = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=TextDataset.collate)

epochs = 100

if savepath.is_file():
    try:
        with savepath.open("rb") as fp:
            model = torch.load(fp)
    except Exception:
        module = sa.SaModel1()
        model = sa.Model(model=module, LossFunction=nn.functional.cross_entropy)
else:
    exit(0)



