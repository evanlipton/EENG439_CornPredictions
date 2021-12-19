import torch

# @Fix


class DataLoaderHelper(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.labels[index]

    def __len__(self):
        return self.data.shape[0]
