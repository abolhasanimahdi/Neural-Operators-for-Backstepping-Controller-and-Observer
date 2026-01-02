import numpy as np
from PDE.pde_model import PDE
from PDE.pde_solver import solve_pde
from PDE.pde_kernel import compute_kernel
from plot import plot_pde, plot_kernel

# Define PDE
pde = PDE()
# Solve PDE u(x,t)
u = solve_pde(pde)
# Compute kernel k(x,y)
k = compute_kernel(pde)

# Time grid
t_grid = np.linspace(0, pde.T, pde.Nt)
# Plot results
plot_pde(pde.x, t_grid, u)
plot_kernel(pde.x, k)
