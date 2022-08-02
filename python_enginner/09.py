'''
epoch = 1 foward & backward pass of ALL training samples

batch_size = number of training samples in 1 forward & backward pass

number of iteration = number of passes, each pass using (batch_size) number of samples

eg.: 100 samples, batch_size=20 --> 100/20 = 5 iteration for 1 epoch
'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# Implement a custom Dataset:
# inherit Dataset
# implement __init__ , __getitem__ , and __len__
class WineDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, 0])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
# print(dataset[0])

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
dataiter = iter(dataloader)

data = dataiter.next()
features, labels = data
# print(features, labels)

# dummy trianing loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
# print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward pass & loss

        # backward pass

        # updates

        # zero gradients

        if (i+1) % 5 ==0:
            print(f'epoch: {epoch+1}/{num_epochs}, step: {i+1}/{n_iterations}, inputs : {inputs.shape}')