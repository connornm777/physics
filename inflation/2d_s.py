import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Parameters
nu = 0.05  # growth rate
D = 0.05  # diffusion coefficient
kx, ky = np.sqrt(nu/D), np.sqrt(nu/D)
Lx, Ly = 10, 10  # domain dimensions
Nx, Ny = 100, 100  # number of spatial points
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
dt = 0.01  # time step
T = 2  # total simulation time

# Spatial grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initial conditions: a Gaussian bump in the center
#S0 = np.exp(-((X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2) / 1)
S0 = 0.1*np.sin(10*kx*X)*np.sin(10*ky*Y)


# Discretize the PDE using finite differences
def solve_pde(S):
    S_next = S.copy()
    for t in np.arange(0, T, dt):
        # Calculate second derivatives with periodic boundary conditions
        S_xx = np.roll(S, -1, axis=0) - 2 * S + np.roll(S, 1, axis=0)
        S_yy = np.roll(S, -1, axis=1) - 2 * S + np.roll(S, 1, axis=1)

        # Update the solution
        S_next += dt * (nu * S + D * (S_xx / dx ** 2 + S_yy / dy ** 2))
        S = S_next
    return S_next


# Set up the figure, axes, and plot element
fig, ax = plt.subplots()
heatmap = ax.imshow(S0, extent=(0, Lx, 0, Ly), origin='lower', cmap='viridis')
cbar = plt.colorbar(heatmap)
cbar.set_label('Concentration')


def init():
    ax.set_title("Time = 0.00 s")
    return (heatmap,)


def animate(i):
    global S0
    S0 = solve_pde(S0)
    heatmap.set_data(S0)
    ax.set_title(f"Time = {i * dt:.2f} s")
    return (heatmap,)


ani = animation.FuncAnimation(fig, animate, frames=int(T / dt), init_func=init, interval=50, repeat=False)

plt.show()