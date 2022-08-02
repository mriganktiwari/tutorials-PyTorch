# MNIST
# DataLoader, Transformation
# Multilayer Neural Net,, activation functions
# Loss and optimizer
# Training loos (batch training)
# Model evaliation
# GPU support

from random import triangular
from turtle import down
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hparams
input_size = 784 # 28x28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
lr = 1e-3

# data
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

# visualize data images
for i in range(6):
    plt.subplot(2,3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out

model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

# loss and optim
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # reshape our images 1st
        # 100x1x28x28 ------> 100x784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')

# testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:  
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # values, index
        _, predictions = torch.max(outputs, dim=1) # gives the index of max probablity class
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    
    print(f'n_samples = {n_samples}, n_correct = {n_correct}')
    acc = 100 * n_correct / n_samples
    print(f'accuracy = {acc}')
    