# training.py
import torch
import deepxde as dde
import numpy as np

def train_operator(x, lam_dataset, k_dataset, samples, nx, branch_layers=[16, 8], trunk_layers=[8, 16], p=16, iterations=500):
    X_mesh, Y_mesh = np.meshgrid(x, x)
    X_branch = np.stack([lam_dataset.astype(np.float32)] * nx, axis=1).reshape(samples, -1)
    X_trunk = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T.astype(np.float32)
    Y = k_dataset.reshape(samples, -1).astype(np.float32)
    idx = int(samples * 0.9)
    dataset = dde.data.TripleCartesianProd(X_train=(X_branch[:idx], X_trunk), y_train=Y[:idx], X_test=(X_branch[idx:], X_trunk), y_test=Y[idx:])
    trunk_net_sizes = [2] + trunk_layers + [p]
    branch_net_sizes = [nx * nx] + branch_layers + [p]
    net = dde.nn.DeepONetCartesianProd(layer_sizes_branch=branch_net_sizes, layer_sizes_trunk=trunk_net_sizes, activation="relu", kernel_initializer="Glorot normal")
    model = dde.Model(dataset, net)
    model.compile(optimizer="adam", lr=1e-3, metrics=["mean l2 relative error"], decay=("step", 1000, 0.9))
    model.train(iterations=iterations, display_every=100)
    return model

def load_operator(nx, branch_layers=[16, 8], trunk_layers=[8, 16], p=16, path="Operator.pth"):
    trunk_net_sizes = [2] + trunk_layers + [p]
    branch_net_sizes = [nx * nx] + branch_layers + [p]
    net = dde.nn.DeepONetCartesianProd(layer_sizes_branch=branch_net_sizes, layer_sizes_trunk=trunk_net_sizes, activation="relu", kernel_initializer="Glorot normal")
    model = dde.Model(None, net)
    model.compile("adam", lr=1e-3)
    checkpoint = torch.load(path, map_location="cpu")
    model.net.load_state_dict(checkpoint["model_state_dict"])
    return model

def compute_k_hat(lam, model, nx):
    x = np.linspace(0, 1, nx).astype(np.float32)
    X_mesh, Y_mesh = np.meshgrid(x, x)
    X_trunk = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T.astype(np.float32)
    lambda_1d = lam(x).astype(np.float32)
    lambda_2d = np.stack([lambda_1d] * nx, axis=0)
    lambda_test = lambda_2d.reshape(1, -1)
    k_hat_flat = model.predict((lambda_test, X_trunk))
    k_hat = np.tril(k_hat_flat.reshape(nx, nx))
    return k_hat

def compute_k_hats(lam_dataset, model, nx, samples):
    x = np.linspace(0, 1, nx).astype(np.float32)
    X_mesh, Y_mesh = np.meshgrid(x, x)
    branch_input = np.stack([lam_dataset] * nx, axis=1).reshape(samples, -1).astype(np.float32)
    trunk_input = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T.astype(np.float32)
    _ = model.predict((branch_input, trunk_input))