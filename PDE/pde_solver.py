import numpy as np

def solve_pde(pde):
    Nx = pde.Nx
    Nt = pde.Nt
    h = pde.h
    dt = pde.dt
    lam = pde.lambda_func(pde.x)
    u0t = pde.u0t_func
    u1t = pde.u1t_func
    # Solution array
    u = np.zeros((Nt, Nx))

    # Initial condition u(x,0)
    u[0, :] = pde.ux0_func()

    for n in range(Nt - 1):
        for i in range(1, Nx - 1):
            u[n + 1, i] = (
                u[n, i] + dt * (
                    (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]) / h**2
                    + lam[i] * u[n, i]
                )
            )

        # Boundary conditions
        t_next = (n + 1) * dt
        u[n + 1, 0] = u0t(t_next)   # Left boundary
        u[n + 1, -1] = u1t(t_next)  # Right boundary
    return u