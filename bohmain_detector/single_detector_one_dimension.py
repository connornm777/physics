import numpy as np
import matplotlib.pyplot as plt

# Constants
m = 1.0  # Mass
r = 0.0  # Rate for g(x)
V_i = 0.0  # Potential V_i
V_f = 0.0  # Potential V_f
dx = 0.1  # Spatial step size
dt = 0.01  # Time step size
t_max = 10.0  # Maximum time
x_min = -10.0  # Minimum x
x_max = 10.0  # Maximum x
epsilon = 1e-8  # Small constant to prevent division by zero

# Spatial and temporal grids
x = np.arange(x_min, x_max + dx, dx)
t = np.arange(0, t_max + dt, dt)
Nx = len(x)
Nt = len(t)

# Initializing variables
rho_i = np.exp(-((x - 3.0) ** 2) / (2 * 1.0 ** 2))
rho_i /= np.sum(rho_i) * dx  # Normalize
rho_f = np.zeros(Nx)
phi_i = 0.1*x # np.zeros(Nx)
phi_f = np.zeros(Nx)

# Function g(x)
g = r * np.exp(-x ** 2 / (2 * 1.0 ** 2))

# Preallocate arrays to store results
rho_i_history = []
rho_f_history = []


# Finite difference functions
def derivative(f):
    df = np.zeros_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
    df[0] = (f[1] - f[0]) / dx  # Forward difference at the left boundary
    df[-1] = (f[-1] - f[-2]) / dx  # Backward difference at the right boundary
    return df


def second_derivative(f):
    ddf = np.zeros_like(f)
    ddf[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / dx ** 2
    ddf[0] = (f[2] - 2 * f[1] + f[0]) / dx ** 2  # Second order at the left boundary
    ddf[-1] = (f[-1] - 2 * f[-2] + f[-3]) / dx ** 2  # Second order at the right boundary
    return ddf


# Time evolution
for n in range(Nt):
    # Store the densities for visualization
    if n % 100 == 0:
        rho_i_history.append(rho_i.copy())
        rho_f_history.append(rho_f.copy())

    # Compute derivatives
    phi_i_x = derivative(phi_i)
    phi_f_x = derivative(phi_f)

    # Compute terms involving rho_i
    rho_i_x = derivative(rho_i)
    rho_i_xx = second_derivative(rho_i)
    rho_i_safe = rho_i + epsilon  # Avoid division by zero
    rho_i_ratio = rho_i_x / rho_i_safe
    quantum_potential_i = (rho_i_xx / rho_i_safe) - 0.5 * (rho_i_ratio) ** 2

    # Compute terms involving rho_f
    rho_f_x = derivative(rho_f)
    rho_f_xx = second_derivative(rho_f)
    rho_f_safe = rho_f + epsilon  # Avoid division by zero
    rho_f_ratio = rho_f_x / rho_f_safe
    quantum_potential_f = (rho_f_xx / rho_f_safe) - 0.5 * (rho_f_ratio) ** 2

    # Update phi_i and phi_f
    phi_i_t = (-1 / (2 * m)) * (phi_i_x) ** 2 - V_i + (1 / (4 * m)) * quantum_potential_i - (phi_f - phi_i) * g
    phi_f_t = (-1 / (2 * m)) * (phi_f_x) ** 2 - V_f + (1 / (4 * m)) * quantum_potential_f
    phi_i += phi_i_t * dt
    phi_f += phi_f_t * dt

    # Recompute derivatives after updating phi
    phi_i_x = derivative(phi_i)
    phi_f_x = derivative(phi_f)

    # Update rho_i and rho_f
    rho_i_flux = phi_i_x * rho_i
    rho_f_flux = phi_f_x * rho_f
    rho_i_t = - (1 / (2 * m)) * derivative(rho_i_flux) - g * rho_i
    rho_f_t = - (1 / (2 * m)) * derivative(rho_f_flux) + g * rho_i
    rho_i += rho_i_t * dt
    rho_f += rho_f_t * dt

    # Ensure non-negativity
    rho_i = np.maximum(rho_i, 0)
    rho_f = np.maximum(rho_f, 0)

    # Normalize densities
    total_density = rho_i + rho_f
    normalization_factor = np.sum(total_density) * dx
    rho_i /= normalization_factor
    rho_f /= normalization_factor

    # Error checking
    if np.any(np.isnan(rho_i)) or np.any(np.isinf(rho_i)):
        print(f"NaN or Inf detected in rho_i at time step {n}")
        break
    if np.any(np.isnan(rho_f)) or np.any(np.isinf(rho_f)):
        print(f"NaN or Inf detected in rho_f at time step {n}")
        break
    if np.any(np.isnan(phi_i)) or np.any(np.isinf(phi_i)):
        print(f"NaN or Inf detected in phi_i at time step {n}")
        break
    if np.any(np.isnan(phi_f)) or np.any(np.isinf(phi_f)):
        print(f"NaN or Inf detected in phi_f at time step {n}")
        break

# Visualization
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
line1, = ax.plot([], [], label=r'$\rho_i$')
line2, = ax.plot([], [], label=r'$\rho_f$')
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, max(np.max(rho_i_history[0]), np.max(rho_f_history[0])) * 1.1)
ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.legend()


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2


def animate(i):
    line1.set_data(x, rho_i_history[i])
    line2.set_data(x, rho_f_history[i])
    return line1, line2


ani = FuncAnimation(fig, animate, frames=len(rho_i_history), init_func=init, blit=True, interval=50)
plt.show()
