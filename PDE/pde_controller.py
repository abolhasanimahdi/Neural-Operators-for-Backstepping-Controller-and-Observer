# pde_controller.py
import numpy as np

def solve_kernel(pde):
    Nx = pde.Nx
    h = pde.h
    x = pde.x
    lam = pde.lambda_func(x)
    k = np.zeros((Nx, Nx))
    for i in range(0, Nx - 1):
        k[i + 1, i + 1] = k[i, i] - (h / 4.0) * (lam[i] + lam[i + 1])
        k[i + 1, i] = k[i, i] - (h / 4.0) * lam[i]
        k[i + 1, 0] = 0
        for j in range(1, i):
            term_reaction = (lam[j] * h ** 2) * (k[i, j + 1] + k[i, j - 1]) / 2.0
            k[i + 1, j] = k[i, j + 1] + k[i, j - 1] - k[i - 1, j] + term_reaction
    return k

def compute_control(k, u, h):
    return h * np.dot(k[-1,:], u)