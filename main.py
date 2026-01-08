# main.py
import numpy as np
from PDE.pde_model import PDE
from PDE.pde_solver import solve_pde
from PDE.pde_controller import solve_kernel
from plot import plot_pde, plot_kernel, plot_lambda, plot_error_norm
import matplotlib.pyplot as plt

# Define PDE
def lambda_func2(x): return 50 * np.cos(8 * np.arccos(x))
pde2 = PDE(lambda_func=lambda_func2)
t_grid2 = np.linspace(0, pde2.T, pde2.Nt)

def lambda_func3(x): return 20 * np.cos(5 * np.arccos(x))
def u1t_func3(t): return 10 * np.cos(np.pi * 2 * t) + 7 * np.sin(16 * t)
pde3 = PDE(lambda_func=lambda_func3, u1t_func=u1t_func3)
t_grid3 = np.linspace(0, pde3.T, pde3.Nt)
# Solve PDE u(x,t)
u2, _ = solve_pde(pde2, k_ctrl=None, k_obs=None)
u3, _ = solve_pde(pde3, k_ctrl=None, k_obs=None)
# Compute kernel k(x,y)
k2 = solve_kernel(pde2)
k3 = solve_kernel(pde3)
# Solve controlled PDE u(x,t)
u2_cl, _ = solve_pde(pde2, k_ctrl=k2, k_obs=None)
u3_cl, _ = solve_pde(pde3, k_ctrl=k3, k_obs=None)

# _, u2_hat = solve_pde(pde2, k_ctrl=None, k_obs=k2[-1, :])
_, u3_hat = solve_pde(pde3, k_ctrl=None, k_obs=k3[-1, :])

error = u3 - u3_hat
# Plot results
plot_pde(pde3.x, t_grid3, u3, title=r'PDE solution $u(x,t)$ for $\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$')
plot_kernel(pde2.x, k2, title=r'Kernel solution $k(x,y)$ for $\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$')
plot_lambda(pde2.x, lambda_func2, title=r'$\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$')
plot_pde(pde3.x, t_grid3, u3_cl, title=r'Controlled PDE solution $u(x,t)$ for $\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$')
plot_pde(pde3.x, t_grid3, u3_hat, title=r'Controlled PDE solution $u(x,t)$ for $\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$')
plot_error_norm(t_grid3, error)
plt.show()