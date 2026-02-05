# pde_controller.py
import numpy as np

def solve_kernel(lambda_func, nx):
    h = 1 / (nx - 1)
    x = np.linspace(0, 1, nx)  
    lam = lambda_func(x)
    k = np.zeros((nx, nx))
    for i in range(0, nx - 1):
        k[i + 1, i + 1] = k[i, i] - (h / 4.0) * (lam[i] + lam[i + 1])
        k[i + 1, i] = k[i, i] - (h / 4.0) * lam[i]
        k[i + 1, 0] = 0
        for j in range(1, i):
            term_reaction = (lam[j] * h ** 2) * (k[i, j + 1] + k[i, j - 1]) / 2.0
            k[i + 1, j] = k[i, j + 1] + k[i, j - 1] - k[i - 1, j] + term_reaction
    return k

def compute_control(k, u, nx):
    h = 1 / (nx - 1)
    return h * np.dot(k[-1,:], u)