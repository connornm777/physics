import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
nu = 0.01  # growth rate
D = 0.5  # diffusion coefficient
L = 10    # length of the domain
N = 100   # number of spatial points
k = np.sqrt(nu/D)
dx = L / (N - 1)
dt = 0.01 # time step
T = 2     # total time of the simulation

# Spatial grid
x = np.linspace(0, L, N)

# Initial conditions
initial_conditions = [np.sin(k*x), np.exp(-((x - L/2)**2) / (0.1 * L**2))]

# Discretize the PDE using finite differences
def solve_pde(S0):
    S = S0.copy()
    for t in np.arange(0, T, dt):
        S_xx = np.roll(S, 1) - 2 * S + np.roll(S, -1)
        S_xx[0] = S_xx[-1] = 0  # Neumann boundary conditions (no flux at boundaries)
        S = S + dt * (nu * S + D * (S_xx / dx**2))
    return S

# Set up the figure and animation
fig, ax = plt.subplots()

lines = []
for ic in initial_conditions:
    line, = ax.plot(x, ic, label='t=0')
    lines.append(line)

def init():
    ax.set_xlim(0, L)
    ax.set_ylim(-1, 1.5)
    return lines

def animate(i):
    for j, line in enumerate(lines):
        S = solve_pde(initial_conditions[j])
        line.set_ydata(S)
        initial_conditions[j] = S
    ax.set_title(f"Time = {i*dt:.2f} s")
    return lines

ani = animation.FuncAnimation(fig, animate, frames=int(T/dt), init_func=init, blit=True, interval=50, repeat=False)

plt.legend()
plt.show()
