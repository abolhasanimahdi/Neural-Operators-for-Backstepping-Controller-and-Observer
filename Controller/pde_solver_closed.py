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

    # Initial condition
    u[0, :] = pde.ux0_func()

    for n in range(Nt - 1):
        u_x1 = (u[n, -1] - u[n, -2]) / h
        integral = np.sum(k[-1, :] * u[n, :] * h)
        U[n] = u_x1 - integral
        u[n, 0] = pde.u0t_func(n * dt)
        u[n, -1] = U[n]
        for i in range(1, Nx - 1):
            u[n + 1, i] = (
                u[n, i]
                + dt * (
                    (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]) / h**2
                    + lam[i] * u[n, i]
                )
            )
        u[n + 1, 0] = pde.u0t_func((n + 1) * dt)
    u_x1 = (u[-1, -1] - u[-1, -2]) / h
    integral = np.sum(k[-1, :] * u[-1, :] * h)
    U[-1] = u_x1 - integral
    return u, U
