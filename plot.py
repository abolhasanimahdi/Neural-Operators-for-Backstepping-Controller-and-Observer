# plot.py
import numpy as np
import matplotlib.pyplot as plt

# Plot PDE solution u(x,t)
def plot_pde(x, t, u, title):
    X, T = np.meshgrid(x, t)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, u)
    x1_line = np.ones_like(t) * x[-1]
    ax.plot(t, x1_line, u[:, -1], zorder=10)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u(x,t)')
    ax.set_xlim(1.5, 0)
    ax.view_init(elev=10, azim=100)
    ax.set_title(title)
    plt.tight_layout()

# Plot kernel solution k(x,y)
def plot_kernel(x, k, title):
    X, Y = np.meshgrid(x, x)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Y, X, k)
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_zlabel('k(x,y)')
    ax.set_title(title)
    ax.set_ylim(1, 0)
    plt.tight_layout()

# Plot lambda(x)
def plot_lambda(x, lambda_func, title):
    y = lambda_func(x)
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel(r'$\lambda(x)$')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()