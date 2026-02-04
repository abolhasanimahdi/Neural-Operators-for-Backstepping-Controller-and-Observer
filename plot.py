# plot.py
import numpy as np
import matplotlib.pyplot as plt

# Plot PDE solution u(x,t)
def plot_pde(x, t, u, title, cols=1):
    if isinstance(u, np.ndarray) and u.ndim == 2:
        u = [u]
    else:
        u = u
    rows = int(np.ceil(len(u) / cols))
    fig = plt.figure(figsize=(cols * 4.5, rows * 3.5))
    for idx, current_u in enumerate(u):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        X, T = np.meshgrid(x, t)
        ax.plot_surface(T, X, current_u)
        x1_line = np.ones_like(t) * x[-1]
        ax.plot(t, x1_line, current_u[:, -1], zorder=10)
        ax.set_xlabel('t', fontsize=10, labelpad=-1)
        ax.set_ylabel('x', fontsize=10, labelpad=-1)
        ax.set_zlabel('u(x,t)', fontsize=10, labelpad=-5)
        ax.xaxis.set_tick_params(labelsize=9, pad=-2)
        ax.yaxis.set_tick_params(labelsize=9, pad=-2)
        ax.zaxis.set_tick_params(labelsize=9, pad=-2)
        ax.set_title(title[idx], y=0.9, fontsize=10)
        ax.set_xlim(t[-1], 0)
        ax.view_init(elev=10, azim=110)
    plt.tight_layout()

# Plot kernel solution k(x,y)
def plot_kernel(x, kernel, title, cols=1):
    if isinstance(kernel, np.ndarray) and kernel.ndim == 2:
        kernel = [kernel]
    else:
        kernel = kernel
    rows = int(np.ceil(len(kernel) / cols))
    fig = plt.figure(figsize=(7, 5) if len(kernel) == 1 else (cols * 4.5, rows * 3))
    for idx, k in enumerate(kernel):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        X, Y = np.meshgrid(x, x)
        ax.plot_surface(Y, X, k)
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_zlabel('k(x,y)')
        ax.set_title(title[idx], y=1, fontsize=12 if len(kernel) == 1 else 10)
        ax.view_init(elev=10, azim=290)
        ax.set_ylim(1, 0)
        ax.set_xlim(0, 1)
    plt.tight_layout()

# Plot λ(x)
def plot_lambda(x, lambda_func, gamma, title):
    if callable(lambda_func):
        lambda_func = lambda_func(x)
    plt.figure(figsize=(4, 3))
    if lambda_func.ndim == 1:
        plt.plot(x, lambda_func, label=f"γ = {gamma:.2f}")
    elif lambda_func.ndim == 2:
        for i in range(lambda_func.shape[0]):
            plt.plot(x, lambda_func[i], label=f"γ = {gamma[i]:.2f}")
    plt.xlabel('x', fontsize=10)
    plt.ylabel(r'$\lambda(x)$', fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.title(title, y=1.02, fontsize=10)
    plt.grid(True)
    plt.legend(fontsize=9)
    plt.xlim(0, 1)
    plt.tight_layout()

def plot_observer_error(t, error, title):
    l2_norm = np.linalg.norm(error, axis=1)
    plt.figure(figsize=(7, 5))
    plt.plot(t, l2_norm, linewidth=2, color='red')
    plt.yscale('log')
    plt.xlabel('Time (t)')
    plt.ylabel(r'$||u(x,t) - \hat{u}(x,t)||_{L^2}$')
    plt.title('Observer Error', y=1)
    plt.title(title, y=1)
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()