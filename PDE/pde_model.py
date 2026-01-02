import numpy as np

class PDE:
    def __init__(self, Nx=101, x_min=0, x_max=1, T=5, dt=1e-6, gamma=5, coefficient=50, initial_value=10):
        self.Nx = Nx
        self.x_min = x_min
        self.x_max = x_max
        self.T = T
        self.dt = dt
        self.x = np.linspace(self.x_min, self.x_max, self.Nx)
        self.dx = self.x[1] - self.x[0]
        self.gamma = gamma
        self.coefficient = coefficient
        self.initial_value = initial_value

    def lambda_x(self):
        return self.coefficient * np.cos(self.gamma * np.arccos(self.x))

    def u0(self):
        return np.full(self.Nx, self.initial_value)

    def bc_left(self, t):
        return 0

    def bc_right(self, t):
        return 0