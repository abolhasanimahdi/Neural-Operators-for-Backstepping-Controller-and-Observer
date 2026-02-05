# training.py
import numpy as np
import deepxde as dde
import torch

def train_deeponet_model(x, lambdas, kernels, n_samples, nx, hidden_size, iterations=500):
    branch_X = lambdas.astype(np.float32)
    X_mesh, Y_mesh = np.meshgrid(x, x)
    trunk_X = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T.astype(np.float32)
    Y = kernels.reshape(n_samples, -1).astype(np.float32)
    idx = int(n_samples * 0.9)
    dataset = dde.data.TripleCartesianProd(X_train=(branch_X[:idx], trunk_X), y_train=Y[:idx], X_test=(branch_X[idx:], trunk_X), y_test=Y[idx:])
    branch_layer_sizes = [nx, hidden_size, hidden_size, hidden_size, hidden_size]
    trunk_layer_sizes = [2, hidden_size, hidden_size, hidden_size, hidden_size]
    net = dde.nn.DeepONetCartesianProd(layer_sizes_branch=branch_layer_sizes, layer_sizes_trunk=trunk_layer_sizes, activation="tanh", kernel_initializer="Glorot uniform")
    model = dde.Model(dataset, net)
    model.compile("adam", lr=1e-3, metrics=["mean l2 relative error"])
    model.train(iterations=iterations, display_every=1000)
    model.compile("L-BFGS")
    model.train()
    return model

def load_deeponet_model(nx, hidden_size, path="Operator.pth"):
    x = np.linspace(0, 1, nx)
    X_mesh, Y_mesh = np.meshgrid(x, x)
    trunk_X = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T.astype(np.float32)
    branch_layer_sizes = [nx, hidden_size, hidden_size, hidden_size, hidden_size]
    trunk_layer_sizes = [2, hidden_size, hidden_size, hidden_size, hidden_size]
    net = dde.nn.DeepONetCartesianProd(layer_sizes_branch=branch_layer_sizes, layer_sizes_trunk=trunk_layer_sizes, activation="tanh", kernel_initializer="Glorot uniform", )
    model = dde.Model(None, net)
    model.compile("adam", lr=1e-3)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.net.load_state_dict(checkpoint["model_state_dict"])
    return model, x, trunk_X

def compute_k_hat(lambda_func, model, x, trunk_X, nx):
    lambda_test = lambda_func(x).reshape(1, -1).astype(np.float32)
    k_hat_flat = model.predict((lambda_test, trunk_X))
    k_hat = np.tril(k_hat_flat.reshape(nx, nx))
    return k_hat