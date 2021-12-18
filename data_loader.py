import torch

class DataLoader(torch.utils.data.Dataset):
    __init__(self, x):
        self.x = x

    __getitem__(self, index):
        return x[index]

    __getitem__(self):
        return len(x)
