# generate_data.py
import numpy as np
from PDE.pde_controller import solve_kernel

def generate_lambda_dataset(amplitude=50, samples=1000, nx=51, gamma_min=4, gamma_max=9, seed=5):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, nx)
    gammas = rng.uniform(gamma_min, gamma_max, size=samples)
    lam_dataset = np.zeros((samples, nx))
    for i, gamma in enumerate(gammas):
        lam_dataset[i] = amplitude * np.cos(gamma * np.arccos(x))
    return x, lam_dataset, gammas

def generate_kernel_dataset(lam_dataset):
    k_dataset = []
    for i in range(len(lam_dataset)):
        lam_i = lam_dataset[i]
        def lam(x, lam=lam_i):
            return lam
        k = solve_kernel(lam=lam, nx=len(lam_i))
        k_dataset.append(k)
    return np.array(k_dataset)