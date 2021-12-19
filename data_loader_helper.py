import torch

# @Fix


class TrainLoaderHelper(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.labels[index]

    def __len__(self):
        return self.data.shape[0]

class TestLoaderHelper(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform, extra = None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.extra = extra

    def __getitem__(self, index):
        if self.extra is None:
            return self.transform(self.data[index]), self.labels[index]
        else:
            return self.transform(self.data[index]), self.labels[index], self.extra[index]

    def __len__(self):
        return self.data.shape[0]
