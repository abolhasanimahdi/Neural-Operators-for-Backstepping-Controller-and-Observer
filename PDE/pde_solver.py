# pde_solver.py
import numpy as np
from PDE.pde_controller import compute_control

def solve_pde(pde, u_hatx0=10, k=None, l=None):
    nx, nt, dt = pde.nx, pde.nt, pde.dt
    x = pde.x
    h = 1 / (nx - 1)
    lam = pde.lam(x)
    u = np.zeros((nt, nx))
    u_hat = np.zeros((nt, nx))
    u[0, :] = pde.ux0_func(x)
    u_hat[0, :] = u_hatx0
    for n in range(nt - 1):
        if k is not None:
            state_for_control = u_hat[n, :] if l is not None else u[n, :]
            U_t = compute_control(k, state_for_control, nx)
        else:
            U_t = pde.u1t_func(n * dt)
        diff_u = (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2]) / h ** 2
        react_u = lam[1:-1] * u[n, 1:-1]
        u[n + 1, 1:-1] = u[n, 1:-1] + dt * (diff_u + react_u)
        u[n + 1, 0] = pde.u0t_func((n + 1) * dt)
        u[n + 1, -1] = U_t
        if l is not None:
            ux_1 = (u[n, -1] - u[n, -2]) / h
            u_hat_x_1 = (u_hat[n, -1] - u_hat[n, -2]) / h
            meas_error = ux_1 - u_hat_x_1
            diff_hat = (u_hat[n, 2:] - 2 * u_hat[n, 1:-1] + u_hat[n, :-2]) / h ** 2
            react_hat = lam[1:-1] * u_hat[n, 1:-1]
            gain_term = l[1:-1] * meas_error
            u_hat[n + 1, 1:-1] = u_hat[n, 1:-1] + dt * (diff_hat + react_hat + gain_term)
            u_hat[n + 1, 0] = 0
            u_hat[n + 1, -1] = U_t
    return u, u_hat