# main.py
import numpy as np
from PDE.pde_model import PDE
from PDE.pde_solver import solve_pde
from PDE.pde_controller import solve_kernel
from plot import plot_pde, plot_kernel, plot_lambda, plot_observer_error
import matplotlib.pyplot as plt

# Define PDE for λ(x) = 50cos(5ccos^-1(x))
def lambda_func1(x): return 50 * np.cos(5 * np.arccos(x))
pde1 = PDE(lambda_func=lambda_func1)
t_grid1 = np.linspace(0, pde1.T, pde1.Nt)

# Define PDE for λ(x) = 50cos(8ccos^-1(x))
def lambda_func2(x): return 50 * np.cos(8 * np.arccos(x))
pde2 = PDE(lambda_func=lambda_func2)
t_grid2 = np.linspace(0, pde2.T, pde2.Nt)

# Define PDE for λ(x) = 20cos(5ccos^-1(x)) and U(t) = 10cos(2πt)+7sin(16t)
def lambda_func3(x): return 20 * np.cos(5 * np.arccos(x))
def u1t_func3(t): return 10 * np.cos(np.pi * 2 * t) + 7 * np.sin(16 * t)
pde3 = PDE(lambda_func=lambda_func3, u1t_func=u1t_func3)
t_grid3 = np.linspace(0, pde3.T, pde3.Nt)

# Define PDE for λ(x) = 20cos(5ccos^-1(x))
def lambda_func4(x): return 20 * np.cos(5 * np.arccos(x))
pde4 = PDE(lambda_func=lambda_func4)
t_grid4 = np.linspace(0, pde4.T, pde4.Nt)

# Solve open loop PDEs u(x,t)
u1, _ = solve_pde(pde1)
u2, _ = solve_pde(pde2)
u3, _ = solve_pde(pde3)
u4, _ = solve_pde(pde4)

# Compute kernel k(x,y)
k1 = solve_kernel(pde1)
k2 = solve_kernel(pde2)
k3 = solve_kernel(pde3)
k4 = solve_kernel(pde4)

# Solve close loop PDE u(x,t)
u1_cl, _ = solve_pde(pde1, k=k1)
u2_cl, _ = solve_pde(pde2, k=k2)
u4_cl, _ = solve_pde(pde4, k=k4)

# Solve open loop observed PDE u_hat(x,t)
_, u3_hat = solve_pde(pde3, u_hatx0=20, l=k3[-1, :])
error3= u3 - u3_hat

# Solve close loop observed PDE u_hat(x,t)
_, u4_hat = solve_pde(pde4, u_hatx0=20, k=k4, l=k4[-1, :])
error4 = u4_cl - u4_hat

# Plot open loop PDEs u(x,t)
plot_lambda(x=pde1.x, lambda_func=np.vstack([lambda_func1(pde1.x), lambda_func2(pde2.x)]), gamma=np.array([5, 8]), title=r'$\lambda(x)=50\cos\!\left(\gamma\cos^{-1}(x)\right)$')

plot_pde(pde1.x, t_grid1, u1, title=r'PDE solution $u(x,t)$ for $\lambda(x)=50\cos\!\left(5\cos^{-1}(x)\right)$')
plot_pde(pde2.x, t_grid2, u2, title=r'PDE solution $u(x,t)$ for $\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$')
plot_pde(pde3.x, t_grid3, u3, title=r'Open loop PDE solution $u(x,t)$ for $\lambda(x)=20\cos\!\left(8\cos^{-1}(x)\right)$ and $U(t)=10cos(2\pi t)+7sin(16t)$')

# Plot closed loop PDEs u(x,t)
plot_kernel(pde1.x, kernel=[k1, k2], title=[r'Kernel solution $k(x,y)$ for $\lambda(x)=50\cos\!\left(5\cos^{-1}(x)\right)$', r'Kernel solution $k(x,y)$ for $\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$'], cols=2)

plot_pde(pde1.x, t_grid1, u1_cl, title=r'Closed loop PDE solution $u(x,t)$ for $\lambda(x)=50\cos\!\left(5\cos^{-1}(x)\right)$')
plot_pde(pde2.x, t_grid2, u2_cl, title=r'Closed loop PDE solution $u(x,t)$ for $\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$')
plot_pde(pde4.x, t_grid4, u4_cl, title=r'Closed loop PDE solution $u(x,t)$ for $\lambda(x)=20\cos\!\left(8\cos^{-1}(x)\right)$')

# Plot observed PDE u_hat(x,t)
plot_pde(pde3.x, t_grid3, u3_hat, title=r'Open loop observed PDE solution $u(x,t)$ for $\lambda(x)=20\cos\!\left(8\cos^{-1}(x)\right)$ and $U(t)=10cos(2\pi t)+7sin(16t)$')
plot_observer_error(t_grid3, error3, title=r'Closed loop observed error $\|e(t)\|_2$ for $\lambda(x)=20\cos\!\left(8\cos^{-1}(x)\right)$ and $U(t)=10cos(2\pi t)+7sin(16t)$')

# Plot close loop observed PDE u_hat(x,t)
plot_pde(pde4.x, t_grid4, u4_hat, title=r'Closed loop observed PDE solution $u(x,t)$ for $\lambda(x)=20\cos\!\left(8\cos^{-1}(x)\right)$')
plot_observer_error(t_grid4, error4, title=r'Closed loop observed error $\|e(t)\|_2$ for $\lambda(x)=20\cos\!\left(8\cos^{-1}(x)\right)$')

plt.show()