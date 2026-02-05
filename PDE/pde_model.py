# pde_model.py
import numpy as np

class PDE:
    def __init__(self, nx=51, T=1.5, dt=1e-5, lambda_func=None, u0t_func=None, u1t_func=None, ux0_func=None):
        self.nx = nx
        self.dt = dt
        self.nt = int(T / dt)
        self.x = np.linspace(0, 1, nx)
        self.t = np.linspace(0, T, self.nt)
        # λ(x)
        if lambda_func is None:
            def lambda_default(x): return 50 * np.cos(5 * np.arccos(x))
            self.lambda_func = lambda_default
        else:
            self.lambda_func = lambda_func
        # Left boundary u(0,t)
        if u0t_func is None:
            def u0t_default(t): return 0
            self.u0t_func = u0t_default
        else:
            self.u0t_func = u0t_func
        # Right boundary u(1,t) = U(t)
        if u1t_func is None:
            def u1t_default(t): return 0
            self.u1t_func = u1t_default
        else:
            self.u1t_func = u1t_func
        # Initial condition u(x,0)
        if ux0_func is None:
            def ux0_default(x): return 10
            self.ux0_func = ux0_default
        else:
            self.ux0_func = ux0_func