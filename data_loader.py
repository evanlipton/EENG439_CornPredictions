import torch

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, x):
        self.x = x

    def __getitem__(self, index):
        return x[index]

    def __getitem__(self):
        return len(x)
