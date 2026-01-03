import numpy as np

def solve_kernel(pde):
    Nx = pde.Nx
    h = pde.h
    x = pde.x
    lam = pde.lambda_func(pde.x)
    k = np.zeros((Nx, Nx))

    k[1, 1] = -(lam[0] + lam[1]) * h / 4

    for i in range(1, Nx - 1):
        k[i + 1, 0] = 0
        k[i + 1, i + 1] = k[i, i] - h / 4 * (lam[i - 1] + lam[i])
        k[i + 1, i] = k[i, i] - h / 2 * lam[i]

        for j in range(1, i):
            k[i + 1, j] = -k[i - 1, j] + k[i, j + 1] + k[i, j - 1] + lam[j] * h ** 2 * (k[i, j + 1] + k[i, j - 1]) / 2
    return k
