import numpy as np
from PDE.pde_model import PDE
from PDE.pde_solver import solve_pde
from PDE.pde_kernel import compute_kernel
from plot import plot_pde, plot_kernel
import matplotlib.pyplot as plt

# Define PDE
def lambda_func1(x): return 50 * np.cos(5 * np.arccos(x))
def lambda_func2(x): return 50 * np.cos(8 * np.arccos(x))
pde1 = PDE(lambda_func=lambda_func1)
pde2 = PDE(lambda_func=lambda_func2)
# Solve PDE u(x,t)
u1 = solve_pde(pde1)
u2 = solve_pde(pde2)
# Compute kernel k(x,y)
k1 = compute_kernel(pde1)
k2 = compute_kernel(pde2)
# Time grid
t_grid1 = np.linspace(0, pde1.T, pde1.Nt)
t_grid2 = np.linspace(0, pde2.T, pde2.Nt)
# Plot results
plot_pde(pde1.x, t_grid1, u1, title=r'PDE solution $u(x,t)$ for $\lambda(x)=50\cos\!\left(5\cos^{-1}(x)\right)$')
plot_kernel(pde1.x, k1, title=r'Kernel solution $k(x,y)$ for $\lambda(x)=50\cos\!\left(5\cos^{-1}(x)\right)$')
plot_pde(pde2.x, t_grid2, u2, title=r'PDE solution $u(x,t)$ for $\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$')
plot_kernel(pde2.x, k2, title=r'Kernel solution $k(x,y)$ for $\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$')
plt.show()