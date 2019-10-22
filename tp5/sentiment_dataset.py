import torch
import gzip


def load_dataset():
    datafile = "train-1000.pth"

    with gzip.open(datafile, "rb") as fp:
        dataset = torch.load(fp)
        print(dataset)

    return dataset
