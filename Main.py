import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

# define the range of x values
x_min = 0.01  # avoid zero division
x_max = 10  # approximate infinity
x = torch.linspace(x_min, x_max, 100, requires_grad=True)
print(x.shape)
x = x.reshape(-1, 1)
print(x.shape)

# define mlp with nn and tanh function and three layer
mlp = nn.Sequential(
    nn.Linear(1, 40),
    nn.Tanh(),
    nn.Linear(40, 30),
    nn.Tanh(),
    nn.Linear(30, 40),
    nn.Tanh(),
    nn.Linear(40, 1),
)

optimizer = optim.SGD(list(mlp.parameters()), lr=0.01, momentum=0.9)
criterion = nn.MSELoss()


# def dx_dy(y, x):
#     return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
#
#
# def d2x_dy2(y, x):
#     return torch.autograd.grad(dx_dy(y, x), x, grad_outputs=torch.ones_like(x), create_graph=True)[0]


def dx_dy(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


# Inside your d2x_dy2 function
def d2x_dy2(y, x):
    dy_dx = dx_dy(y, x)
    return torch.autograd.grad(dy_dx, x, grad_outputs=torch.ones_like(dy_dx), create_graph=True)[0]


losses = []

for i in range(1500):
    y = mlp.forward(x)
    y_p = dx_dy(y, x)
    y_pp = d2x_dy2(y, x)

    # modify the equation to match the one you sent
    residential = x * y_pp - y ** 3
    initial = y[0] - 1
    final = y[-1]  # approximate the limit as x approaches infinity

    # modify the loss function to include the final condition
    loss = (residential ** 2).mean() + initial ** 2 + final ** 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.detach().numpy())

    if i % 10 == 0:
        print(i, loss.detach().numpy()[0])

# write a test for function
x_test = torch.linspace(x_min, x_max, 31).reshape(-1, 1)
# there is no exact solution for this equation, so we use the mlp output as the prediction
predict = mlp.forward(x_test).detach().numpy()

# write a test for function and calculate the derivative for x = 0
# Ensure that x and y tensors have requires_grad=True
x_zero = torch.tensor([[0.0]], requires_grad=True)
y_zero = mlp.forward(x_zero)
derivative_at_zero = dx_dy(y_zero, x_zero)
print(f"Derivative at x = 0: {derivative_at_zero.item()}")

# plot the prediction
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(x_test, predict, "r.", label='predict')
axs[1].plot(np.log10(losses), "c", label='loss')

for ax in axs:
    ax.legend()
    ax.grid()

plt.show()
