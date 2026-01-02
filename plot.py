import numpy as np
import matplotlib.pyplot as plt

# Plot PDE solution u(x,t)
def plot_pde(x, t, u):
    X, T = np.meshgrid(x, t)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, u)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    ax.set_title('PDE solution u(x,t)')
    plt.tight_layout()
    plt.show()

# Plot kernel solution k(x,y)
def plot_kernel(x, k):
    X, Y = np.meshgrid(x, x)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, k)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('k(x,y)')
    ax.set_title('Kernel solution k(x,y)')
    plt.tight_layout()
    plt.show()