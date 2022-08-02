# 1- Design model (input, output size, forward pass)
# 2- Construct loss and optim
# 3- Training loop (iterate the below steps)
#   - forward pass
#   - backward pass
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0- prepare data
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

print("x, y shapes: --> ", x.shape, y.shape)

y = y.view(y.shape[0], 1)
print("y after reshape: -------> ", y.shape)

n_samples, n_features = x.shape

# 1- model
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

# 2- loss and optimizer
lr = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 3- training loop
num_epochs = 500
for epoch in range(num_epochs):
    # forward pass & loss
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    # zero the gradients
    optimizer.zero_grad()
    if (epoch+1) %10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# plot
predicted = model(x).detach() # to make require_grad False
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show()

