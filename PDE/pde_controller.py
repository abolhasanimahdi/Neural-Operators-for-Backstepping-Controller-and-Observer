# pde_controller.py
import numpy as np

def solve_kernel(pde):
    Nx = pde.Nx
    h = pde.h
    x = pde.x
    lam = pde.lambda_func(x)
    k = np.zeros((Nx, Nx))

    for i in range(1, Nx - 1):
        k[i + 1, 0] = 0
        k[i + 1, i + 1] = k[i, i] - h / 4 * (lam[i - 1] + lam[i])
        k[i + 1, i] = k[i, i] - h / 2 * lam[i]

        for j in range(1, i):
            k[i + 1, j] = -k[i - 1, j] + k[i, j + 1] + k[i, j - 1] + lam[j] * h ** 2 * (k[i, j + 1] + k[i, j - 1]) / 2
    return k

def compute_control(k, current_u, h):
    integrand = k[-1, :] * current_u
    total_sum = 0.5 * integrand[0] + np.sum(integrand[1:-1]) + 0.5 * integrand[-1]
    return h * total_sum