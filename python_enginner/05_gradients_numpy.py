import numpy as np

# f = w * x

# f = 2 * x
x = np.array([1,2,3,4], dtype=np.float32)
y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

# model prediction
def forward(x):
    ''' model prediction '''
    return w * x

# loss = MSE
def loss(y, y_predicted):
    ''' loss calculation '''

    return ((y - y_predicted)**2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x * (w*x - y)
def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
lr = 0.01
iters = 20

for epoch in range(iters):
    # prediction
    y_pred = forward(x)

    # loss
    l = loss(y, y_pred)

    # gradients
    dw = gradient(x,y,y_pred)

    # update weights
    w -= lr * dw

    if epoch % 1 == 0:
        print(f'epoch {epoch}: w = {w:.8f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')


