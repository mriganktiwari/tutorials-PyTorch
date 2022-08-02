from asyncio import as_completed
from email.mime import image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hparams
num_epochs = 5
batch_size = 4
lr = 1e-3

# data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean of 3 channels, and std of 3 channels provided

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# print(type(classes))

# show images
def imshow(img):
    img = img/2 + 0.5 # un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = dataiter.next()

# imshow(torchvision.utils.make_grid(images))

# # cnn test - why fc1 in ConvNet has 16*5*5 input size
# # COMMENT this part when working on the model
# conv1 = nn.Conv2d(3,6,5)
# pool = nn.MaxPool2d(2,2)
# conv2 = nn.Conv2d(6,16,5)
# print(images.shape) # torch.Size([4, 3, 32, 32])
# x=conv1(images)
# print(f'Shape of tensor from conv1 ---> {x.shape}') # torch.Size([4, 6, 28, 28])
# x = pool(x)
# print(f'Shape of tensor from pool ---> {x.shape}') # torch.Size([4, 6, 14, 14])
# x = conv2(x)
# print(f'Shape of tensor from conv2 ---> {x.shape}') # torch.Size([4, 16, 10, 10])
# x = pool(x)
# print(f'Shape of tensor from pool ---> {x.shape}') # torch.Size([4, 16, 5, 5])
# # As seen the tensor output from conv2-pool is 16 channels of 5x5 - when we flatten them for fully connected layers
# # COMMENT this part above when working on the model

# conv net
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'epoch [{epoch+1} / {num_epochs}], Step [{i+1} / {n_total_steps}], Loss: {loss.item():.4f}')

print("Finished training")
path = './cnn.pth'
torch.save(model.state_dict(), path)

# testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max returns (value, index)
        _, predicted = torch.max(outputs, dim=1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
    
    acc = 100 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
