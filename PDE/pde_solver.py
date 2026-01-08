# pde_solver.py
import numpy as np
from PDE.pde_controller import compute_control

def solve_pde(pde, k=None):
    Nx, Nt = pde.Nx, pde.Nt
    h, dt = pde.h, pde.dt
    x = pde.x
    lam = pde.lambda_func(x)
    u = np.zeros((Nt, Nx))
    u[0, :] = pde.ux0_func(x)
    if k is not None:
        u0_control = compute_control(k, u[0, :], h)
        u[0, -1] = u0_control
    else:
        u[0, -1] = 0
    for n in range(Nt - 1):
        diffusion = (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2]) / h ** 2
        reaction = lam[1:-1] * u[n, 1:-1]
        u[n + 1, 1:-1] = u[n, 1:-1] + dt * (diffusion + reaction)
        u[n + 1, 0] = pde.u0t_func((n + 1) * dt)
        if k is not None:
            u[n + 1, -1] = compute_control(k, u[n + 1, :], h)
        else:
            u[n + 1, -1] = pde.u1t_func((n + 1) * dt)
    return u