import numpy as np
import deepxde as dde
from deepxde.data import TripleCartesianProd
from deepxde.nn import DeepONetCartesianProd
from generate_data import generate_lambda_dataset, generate_kernel_dataset

nx = 21
n_samples = 20
hidden_size = 16

x_grid, lambdas, gammas = generate_lambda_dataset(n_samples=n_samples, nx=nx)
kernels = generate_kernel_dataset(lambdas)

branch_X = lambdas.astype(np.float32)
X_mesh, Y_mesh = np.meshgrid(x_grid, x_grid)
trunk_X = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T.astype(np.float32)
Y = kernels.reshape(n_samples, -1).astype(np.float32)

idx = int(n_samples * 0.8)
dataset = TripleCartesianProd(X_train=(branch_X[:idx], trunk_X), y_train=Y[:idx], X_test=(branch_X[idx:], trunk_X), y_test=Y[idx:])

branch_layer_sizes = [nx, hidden_size, hidden_size, hidden_size, hidden_size]
trunk_layer_sizes = [2, hidden_size, hidden_size, hidden_size, hidden_size]

net = DeepONetCartesianProd( layer_sizes_branch=branch_layer_sizes, layer_sizes_trunk=trunk_layer_sizes, activation="tanh", kernel_initializer="Glorot uniform")

model = dde.Model(dataset, net)
model.compile("adam", lr=1e-2, metrics=["mean l2 relative error"])
model.train(iterations=500, display_every=500)

model.compile("L-BFGS")
model.train(iterations=1500, display_every=500)

model.save("Operator.pth")