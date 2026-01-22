import numpy as np
import matplotlib.pyplot as plt
from PDE.pde_model import PDE
from PDE.pde_controller import solve_kernel
from plot import plot_kernel, plot_lambda

def generate_lambda_dataset(amplitude=50, n_samples=1000, nx=101, gamma_min=4, gamma_max=9, seed=5):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, nx)
    gammas = rng.uniform(gamma_min, gamma_max, size=n_samples)
    lambdas = np.zeros((n_samples, nx))
    for i, gamma in enumerate(gammas):
        lambdas[i] = amplitude * np.cos(gamma * np.arccos(x))
    return x, lambdas, gammas

x, lambda_dataset, gammas = generate_lambda_dataset(n_samples=5, nx=200)

kernel_dataset = []
for i in range(len(lambda_dataset)):
    lambda_i = lambda_dataset[i]
    def lambda_func(x, lam=lambda_i):
        return lam
    pde = PDE(N=len(lambda_i) - 1, lambda_func=lambda_func)
    k = solve_kernel(pde)
    kernel_dataset.append(k)
kernel_dataset = np.array(kernel_dataset)

plot_kernel(x, kernel_dataset, title=r'Kernel solution $k(x,y)$')

plot_lambda(x, lambda_dataset, gammas, title=r'$\lambda(x)=50\cos\!\left(\gamma\cos^{-1}(x)\right)$')
plt.show()