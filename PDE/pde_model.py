# pde_model.py
import numpy as np

class PDE:
    def __init__(self, nx=101, T=1.5, dt=1e-5, lam=None, u0t=None, u1t=None, ux0=None):
        self.nx = nx
        self.dt = dt
        self.nt = int(T / dt)
        self.x = np.linspace(0, 1, nx)
        self.t = np.linspace(0, T, self.nt)
        # λ(x)
        if lam is None:
            def lam_default(x): return 50 * np.cos(5 * np.arccos(x))
            self.lam = lam_default
        else:
            self.lam = lam
        # Left boundary u(0,t)
        if u0t is None:
            def u0t_default(t): return 0
            self.u0t_func = u0t_default
        else:
            self.u0t_func = u0t
        # Right boundary u(1,t) = U(t)
        if u1t is None:
            def u1t_default(t): return 0
            self.u1t_func = u1t_default
        else:
            self.u1t_func = u1t
        # Initial condition u(x,0)
        if ux0 is None:
            def ux0_default(x): return 10
            self.ux0_func = ux0_default
        else:
            self.ux0_func = ux0