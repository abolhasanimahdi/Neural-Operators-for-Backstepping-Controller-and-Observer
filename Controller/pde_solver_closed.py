# pde_solver_closed.py
import numpy as np

def solve_pde_closed_loop(pde, k):
    Nx = pde.Nx
    Nt = pde.Nt
    h = pde.h
    dt = pde.dt
    x = pde.x
    lam = pde.lambda_func(x)
    u = np.zeros((Nt, Nx))
    U = np.zeros(Nt)

    u[0, :] = pde.ux0_func()

    def get_control(current_u):
        integrand = k[-1, :] * current_u
        total_sum = 0.5 * integrand[0] + np.sum(integrand[1:-1]) + 0.5 * integrand[-1]
        return h * total_sum

    U[0] = get_control(u[0, :])
    u[0, -1] = U[0]

    for n in range(Nt - 1):
        for i in range(1, Nx - 1):
            u[n + 1, i] = (
                    u[n, i]
                    + dt * (
                            (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]) / h ** 2
                            + lam[i] * u[n, i]
                    )
            )
        u[n + 1, 0] = pde.u0t_func((n + 1) * dt)
        U[n + 1] = get_control(u[n + 1, :])
        u[n + 1, -1] = U[n + 1]

    return u, U