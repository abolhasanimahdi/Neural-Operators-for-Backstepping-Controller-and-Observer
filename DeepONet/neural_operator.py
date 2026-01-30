import numpy as np
import torch
import deepxde as dde
from plot import plot_kernel
import matplotlib.pyplot as plt

nx = 51
hidden_size = 32

x_grid = np.linspace(0, 1, nx)
X_mesh, Y_mesh = np.meshgrid(x_grid, x_grid)
trunk_X = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T.astype(np.float32)

branch_layer_sizes = [nx, hidden_size, hidden_size, hidden_size, hidden_size]
trunk_layer_sizes = [2, hidden_size, hidden_size, hidden_size, hidden_size]

net = dde.nn.DeepONetCartesianProd(
    layer_sizes_branch=branch_layer_sizes,
    layer_sizes_trunk=trunk_layer_sizes,
    activation="tanh",
    kernel_initializer="Glorot uniform",
)

model = dde.Model(None, net)
model.compile("adam", lr=1e-3)

checkpoint = torch.load("Operator.pth-18947.pt", map_location=torch.device('cpu'))
model.net.load_state_dict(checkpoint["model_state_dict"])

def test_lambda_func(x):
    return 50 * np.cos(8 * np.arccos(x))

lambda_test = test_lambda_func(x_grid).reshape(1, -1).astype(np.float32)
k_hat_flat = model.predict((lambda_test, trunk_X))
k_hat = np.tril(k_hat_flat.reshape(nx, nx))

plot_kernel(x_grid, kernel=k_hat, title=[r'Kernel solution $\hat{k}(x,y)$ for $\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$'], cols=1)
plt.show()