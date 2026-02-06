# plot.py
import numpy as np
import matplotlib.pyplot as plt

# Plot PDE solution u(x,t)
def plot_pde(x, t, u, fig_title, title, z_title, cols=1):
    if isinstance(u, np.ndarray) and u.ndim == 2:
        u = [u]
    else:
        u = u
    rows = int(np.ceil(len(u) / cols))
    fig = plt.figure(figsize=(cols * 4.5, rows * 3.5))
    fig.suptitle(fig_title, y=0.9, fontsize=12)
    for idx, current_u in enumerate(u):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        X, T = np.meshgrid(x, t)
        ax.plot_surface(T, X, current_u)
        x1_line = np.ones_like(t) * x[-1]
        ax.plot(t, x1_line, current_u[:, -1], zorder=10)
        ax.set_xlabel(r'$t$', fontsize=10, labelpad=-1)
        ax.set_ylabel(r'$x$', fontsize=10, labelpad=-1)
        ax.set_zlabel(z_title[idx], fontsize=10, labelpad=-5)
        ax.xaxis.set_tick_params(labelsize=9, pad=-2)
        ax.yaxis.set_tick_params(labelsize=9, pad=-2)
        ax.zaxis.set_tick_params(labelsize=9, pad=-2)
        ax.set_title(title[idx], y=0.9, fontsize=10)
        ax.set_xlim(t[-1], 0)
        ax.view_init(elev=10, azim=110)
    plt.tight_layout()

# Plot kernel solution k(x,y)
def plot_kernel(x, k, fig_title, title, z_title, cols=1):
    if isinstance(k, np.ndarray) and k.ndim == 2:
        k = [k]
    else:
        k = k
    rows = int(np.ceil(len(k) / cols))
    fig = plt.figure(figsize=(cols * 4.5, rows * 3.5))
    fig.suptitle(fig_title, y=0.9, fontsize=12)
    for idx, k in enumerate(k):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        X, Y = np.meshgrid(x, x)
        ax.plot_surface(Y, X, k)
        ax.set_xlabel(r'$y$', fontsize=10, labelpad=-1)
        ax.set_ylabel(r'$x$', fontsize=10, labelpad=-1)
        ax.set_zlabel(z_title[idx], fontsize=10, labelpad=-3)
        ax.xaxis.set_tick_params(labelsize=9, pad=-2)
        ax.yaxis.set_tick_params(labelsize=9, pad=-2)
        ax.zaxis.set_tick_params(labelsize=9, pad=-2)
        ax.set_title(title[idx], y=0.9, fontsize=10)
        ax.view_init(elev=10, azim=290)
        ax.set_ylim(1, 0)
        ax.set_xlim(0, 1)
    plt.tight_layout()

# Plot λ(x)
def plot_lambda(x, lam, gamma, title):
    if callable(lam):
        lam = lam(x)
    plt.figure(figsize=(4, 3))
    if lam.ndim == 1:
        plt.plot(x, lam, label=f"γ = {gamma:.2f}")
    elif lam.ndim == 2:
        for i in range(lam.shape[0]):
            plt.plot(x, lam[i], label=f"γ = {gamma[i]:.2f}")
    plt.xlabel(r'$x$', fontsize=10)
    plt.ylabel(r'$\lambda(x)$', fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.title(title, y=1.02, fontsize=10)
    plt.grid(True)
    plt.legend(fontsize=9)
    plt.xlim(0, 1)
    plt.tight_layout()

# Plot ║e(t)║_2
def plot_observer_error(t, e, title):
    l2_norm = np.linalg.norm(e, axis=1)
    plt.figure(figsize=(4, 3))
    plt.plot(t, l2_norm, linewidth=2, color='red')
    plt.yscale('log')
    plt.xlabel(r'$t$', fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylabel(r'$||u(x,t) - \hat{u}(x,t)||_{L^2}$')
    plt.title('Observer Error', y=1)
    plt.title(title, y=1.03, fontsize=10)
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()

# Plot U(t)
def plot_U(U, labels, title):
    t = np.linspace(0, 3, 500)
    plt.figure(figsize=(4, 3))
    if callable(U):
        U = [U]
    if labels is None:
        labels = [f"U{i+1}(t)" for i in range(len(U))]
    for Ui, label in zip(U, labels):
        plt.plot(t, Ui(t), label=label)
    plt.xlabel(r"$t$", fontsize=10)
    plt.ylabel(r"$U(t)$", fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.title(title, y=1.02, fontsize=10)
    plt.grid(True)
    plt.legend(fontsize=7, loc='lower center')
    plt.tight_layout()

def plot_learning_error(U1, U1_hat, U2, U2_hat, nx, t):
    e1 = np.sqrt(np.sum((U1 - U1_hat)**2, axis=1) * 1/(nx-1))
    e2 = np.sqrt(np.sum((U2 - U2_hat)**2, axis=1) * 1/(nx-1))
    plt.figure(figsize=(7,3))
    plt.subplot(1,2,1)
    plt.plot(t, e1)
    plt.xlabel("t", fontsize=10)
    plt.ylabel("e(t)", fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlim(-0.1, 1.5)
    plt.title(r'$\sqrt{\sum_{i=0}^{n_x} \left(u[i] - \hat{u}[i]\right)^2  dx}$ for $\lambda(x)=50\cos\!\left(5\cos^{-1}(x)\right)$', fontsize=9)
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(t, e2)
    plt.xlabel("t", fontsize=10)
    plt.ylabel("e(t)", fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlim(-0.1, 1.5)
    plt.title(r'$\sqrt{\sum_{i=0}^{n_x} \left(u[i] - \hat{u}[i]\right)^2  dx}$ for $\lambda(x)=50\cos\!\left(8\cos^{-1}(x)\right)$', fontsize=9)
    plt.grid(True)
    plt.tight_layout()