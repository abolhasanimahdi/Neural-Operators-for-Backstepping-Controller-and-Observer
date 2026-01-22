import numpy as np
import matplotlib.pyplot as plt
from plot import plot_lambda

def generate_lambda_dataset(amplitude=50, n_samples=1000, nx=101, gamma_min=4, gamma_max=9, seed=5):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, nx)
    gammas = rng.uniform(gamma_min, gamma_max, size=n_samples)
    lambdas = np.zeros((n_samples, nx))
    for i, gamma in enumerate(gammas):
        lambdas[i] = amplitude * np.cos(gamma * np.arccos(x))
    return x, lambdas, gammas

x, lambdas, gammas = generate_lambda_dataset(n_samples=5, nx=200)
plot_lambda(x, lambdas, gammas, title=r'$\lambda(x)=50\cos\!\left(\gamma\cos^{-1}(x)\right)$')
plt.show()