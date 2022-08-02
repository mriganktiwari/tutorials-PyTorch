# 1- Design model (input, output size, forward pass)
# 2- Construct loss and optim
# 3- Training loop (iterate the below steps)
#   - forward pass
#   - backward pass
#   - update weights

import torch
import torch.nn as nn

# f = w * x

# f = 2 * x
x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
x_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = x.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
# model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        # define layers
        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

# # model prediction
# def forward(x):
#     ''' model prediction '''
#     return w * x

# # loss = MSE
# def loss(y, y_predicted):
#     ''' loss calculation '''
#     return ((y - y_predicted)**2).mean()

# # gradient
# # MSE = 1/N * (w*x - y)**2
# # dJ/dw = 1/N * 2x * (w*x - y)
# def gradient(x,y,y_predicted):
#     return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training: f(5) = {model(x_test).item():.3f}')

# Training
lr = 0.01
iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(iters):
    # prediction
    y_pred = model(x)

    # loss
    l = loss(y, y_pred)

    # gradients
    # dw = gradient(x,y,y_pred)
    l.backward() # dl/dw

    # update weights
    # with torch.no_grad():
    #     w -= lr * w.grad
    optimizer.step()

    # zero the gradients again
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch}: w = {w[0][0]:.8f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(x_test).item():.3f}')
