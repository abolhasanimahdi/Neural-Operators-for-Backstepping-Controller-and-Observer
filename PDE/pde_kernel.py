import numpy as np

def compute_kernel(pde):
    Nx = pde.Nx
    h = pde.h
    lam = pde.lambda_func(pde.x)
    # Solution array
    k = np.zeros((Nx, Nx))

    for i in range(1, Nx):
        k[i, i] = k[i - 1, i - 1] + (h**2 / 2) * lam[i]

    for i in range(2, Nx):
        for j in range(1, i):
            jp = j + 1 if j + 1 < Nx else j
            jm = j - 1 if j - 1 >= 0 else j
            k[i, j] = (
                -k[i - 2, j]
                + k[i - 1, jp]
                + k[i - 1, jm]
                + (h**2 / 2) * lam[j] * (k[i - 1, jp] + k[i - 1, jm])
            )

    # Boundary on y=0
    k[:, 0] = 0
    return k