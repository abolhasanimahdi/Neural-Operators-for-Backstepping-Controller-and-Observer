import numpy as np
import torch
import deepxde as dde
import matplotlib.pyplot as plt

nx = 21
hidden_size = 16

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

checkpoint = torch.load("Operator.pth-3508.pt", map_location=torch.device('cpu'))
model.net.load_state_dict(checkpoint["model_state_dict"])

def test_lambda_func(x):
    return 50 * np.cos(8 * np.arccos(x))

lambda_test = test_lambda_func(x_grid).reshape(1, -1).astype(np.float32)
k_pred_flat = model.predict((lambda_test, trunk_X))
k_pred = np.tril(k_pred_flat.reshape(nx, nx))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_mesh, Y_mesh, k_pred)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('k(x,y)')
ax.set_title('DeepONet Predicted Kernel $k(x,y)$')
plt.show()