# main.py
import numpy as np
from PDE.pde_model import PDE
from PDE.pde_solver import solve_pde
from Controller.pde_kernel import solve_kernel
from plot import plot_pde, plot_kernel, plot_lambda, plot_controller
import matplotlib.pyplot as plt
from Controller.pde_solver_closed import solve_pde_closed_loop

# Define PDE
def lambda_func2(x): return 50 * np.cos(8 * np.arccos(x))
pde2 = PDE(lambda_func=lambda_func2)
# Time grid
t_grid2 = np.linspace(0, pde2.T, pde2.Nt)
# Solve PDE u(x,t)
u2 = solve_pde(pde2)

# Compute kernel k(x,y)
k2 = solve_kernel(pde2)
# Compute controller U(t) for pde2
u2_cl, U2 = solve_pde_closed_loop(pde2, k2)

# Plot results
plot_pde(pde2.x, t_grid2, u2, title=r'PDE solution $u(x,t)$ for $\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$')
plot_kernel(pde2.x, k2, title=r'Kernel solution $k(x,y)$ for $\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$')
plot_lambda(pde2.x, lambda_func2, title=r'$\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$')
plot_controller(t_grid2, U2, title=r'Controller $U(t)$ for $\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$')
plot_pde(pde2.x, t_grid2, u2_cl, title=r'Controlled PDE solution $u(x,t)$ with backstepping $U(t)$')
plt.show()