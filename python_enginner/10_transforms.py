import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# https://pytorch.org/vision/stable/transforms.html

# Implement a custom Dataset:
# inherit Dataset
# implement __init__ , __getitem__ , and __len__
class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:] # we'll convert into torch.tensor in a new custom Transforms class
        self.y = xy[:, 0] # we'll convert into torch.tensor in a new custom Transforms class
        self.n_samples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample =  self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        print("Type of sample: ----> ", type(sample))
        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(np.asarray(inputs)), torch.from_numpy(np.asarray(targets))

class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))
# print(features.shape, labels.shape)
# print(features, labels)

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))