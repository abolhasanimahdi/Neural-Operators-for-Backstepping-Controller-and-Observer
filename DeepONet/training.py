# training.py
import torch
import deepxde as dde
import numpy as np

class BranchNet(torch.nn.Module):
    def __init__(self, nx, latent_dim):
        super().__init__()
        self.nx = nx
        self.conv1 = torch.nn.Conv2d(1, 16, 5, stride=2)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 32, 5, stride=2)
        self._to_linear = self._get_conv_output_shape()
        self.fc1 = torch.nn.Linear(self._to_linear, 1024)
        self.fc2 = torch.nn.Linear(1024, latent_dim)
    def _get_conv_output_shape(self):
        x = torch.zeros(1, 1, self.nx, self.nx)
        x = self.conv1(x)
        x = self.conv2(x)
        return int(np.prod(x.size()))
    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 1, self.nx, self.nx))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_operator(x, lam_dataset, k_dataset, samples, nx, trunk_layers=[8, 16, 16], iterations=500):
    branch_X_raw = lam_dataset.astype(np.float32)
    branch_X = np.stack([branch_X_raw] * nx, axis=1).reshape(samples, -1)
    X_mesh, Y_mesh = np.meshgrid(x, x)
    trunk_X = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T.astype(np.float32)
    Y = k_dataset.reshape(samples, -1).astype(np.float32)
    idx = int(samples * 0.9)
    dataset = dde.data.TripleCartesianProd(X_train=(branch_X[:idx], trunk_X), y_train=Y[:idx], X_test=(branch_X[idx:], trunk_X), y_test=Y[idx:])
    latent_dim = trunk_layers[-1]
    branch_net = BranchNet(nx, latent_dim)
    trunk_net_sizes = [2] + trunk_layers
    net = dde.nn.DeepONetCartesianProd(layer_sizes_branch=[nx * nx, branch_net], layer_sizes_trunk=trunk_net_sizes, activation="relu", kernel_initializer="Glorot normal")
    model = dde.Model(dataset, net)
    model.compile(optimizer="adam", lr=1e-3, metrics=["mean l2 relative error"])
    model.train(iterations=iterations, display_every=50)
    return model

def load_operator(nx, trunk_layers=[8, 16, 16], path="Operator.pth"):
    latent_dim = trunk_layers[-1]
    branch_net = BranchNet(nx, latent_dim)
    trunk_net_sizes = [2] + trunk_layers
    net = dde.nn.DeepONetCartesianProd(layer_sizes_branch=[nx * nx, branch_net], layer_sizes_trunk=trunk_net_sizes, activation="relu", kernel_initializer="Glorot normal")
    model = dde.Model(None, net)
    model.compile("adam", lr=1e-3)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.net.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.net.load_state_dict(checkpoint)
    return model

def compute_k_hat(lam, model, nx):
    x = np.linspace(0, 1, nx).astype(np.float32)
    X_mesh, Y_mesh = np.meshgrid(x, x)
    trunk_X = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T.astype(np.float32)
    lambda_1d = lam(x).astype(np.float32)
    lambda_2d = np.stack([lambda_1d] * nx, axis=0)
    lambda_test = lambda_2d.reshape(1, -1)
    k_hat_flat = model.predict((lambda_test, trunk_X))
    k_hat = np.tril(k_hat_flat.reshape(nx, nx))
    return k_hat

def compute_k_hats(lam_dataset, model, nx, samples):
    x = np.linspace(0, 1, nx).astype(np.float32)
    X_mesh, Y_mesh = np.meshgrid(x, x)
    branch_input = np.stack([lam_dataset] * nx, axis=1).reshape(samples, -1).astype(np.float32)
    trunk_input = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T.astype(np.float32)
    _ = model.predict((branch_input, trunk_input))