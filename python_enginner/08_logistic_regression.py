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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0- prepare data
# ------------------
bc = datasets.load_breast_cancer()
x,y = bc.data, bc.target
print("x, y shapes --------> ", x.shape, y.shape)

n_samples, n_features = x.shape

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1234)

# scale data - make out data to have 0 mean and variance 1.
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1- model
# f = wx + b, sigmoid at the end
# -------------------------------

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.liniear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.liniear(x))
        return y_pred

model = LogisticRegression(n_features)

# 2- loss & optimizer
lr = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 3- training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass & loss
    y_predicted = model(x_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # updates
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 ==0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(x_test)
    y_predicted_cls = y_predicted.round()

    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy = {acc:.4f}')