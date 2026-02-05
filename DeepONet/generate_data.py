import numpy as np
from PDE.pde_model import PDE
from PDE.pde_controller import solve_kernel

def generate_lambda_dataset(amplitude=50, n_samples=1000, nx=101, gamma_min=4, gamma_max=9, seed=5):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, nx)
    gammas = rng.uniform(gamma_min, gamma_max, size=n_samples)
    lambdas = np.zeros((n_samples, nx))
    for i, gamma in enumerate(gammas):
        lambdas[i] = amplitude * np.cos(gamma * np.arccos(x))
    return x, lambdas, gammas

def generate_kernel_dataset(lambda_dataset):
    kernel_dataset = []
    for i in range(len(lambda_dataset)):
        lambda_i = lambda_dataset[i]
        def lambda_func(x, lam=lambda_i):
            return lam
        k = solve_kernel(lambda_func=lambda_func, nx=len(lambda_i))
        kernel_dataset.append(k)
    return np.array(kernel_dataset)