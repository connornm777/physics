import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the system of ODEs
def system(t, Y):
    x, sigma, x_dot, sigma_dot = Y
    # Avoid division by zero in case sigma is near 0
    if sigma == 0:
        return [x_dot, sigma_dot, 0, 0]
    dxdt = x_dot
    dsigdt = sigma_dot
    dx_dotdt = (2 * sigma_dot * x_dot) / sigma
    dsigma_dotdt = (2*sigma_dot**2 - x_dot**2) / (2*sigma)
    return [dxdt, dsigdt, dx_dotdt, dsigma_dotdt]

# Initial conditions
x0 = 1.0
sigma0 = 10.0
x_dot0 = 1.0
sigma_dot0 = 1.0
Y0 = [x0, sigma0, x_dot0, sigma_dot0]

# Time span for the solution
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(system, t_span, Y0, t_eval=t_eval)

x = sol.y[0]
sigma = sol.y[1]

# ---- Static Plots ----
# Phase plot x vs sigma
plt.figure()
plt.plot(sigma, x)
plt.xlabel('$\sigma$')
plt.ylabel('$x$')
plt.title('Trajectory in (σ, x) space')
plt.grid(True)
plt.show()

# Phase plots for velocities
plt.figure()
plt.plot(x, sol.y[2], label='x vs. dx/dt')
plt.xlabel('$x$')
plt.ylabel('$\dot{x}$')
plt.title('Phase Diagram (x, dx/dt)')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(sigma, sol.y[3], label='σ vs. dσ/dt')
plt.xlabel('$\sigma$')
plt.ylabel('$\dot{\sigma}$')
plt.title('Phase Diagram (σ, dσ/dt)')
plt.grid(True)
plt.show()

# ---- Animation of (σ, x) trajectory ----
fig, ax = plt.subplots()
ax.set_xlim(min(sigma), max(sigma))
ax.set_ylim(min(x), max(x))
ax.set_xlabel('$\sigma$')
ax.set_ylabel('$x$')
ax.set_title('Animation of trajectory in (σ, x) space')
line, = ax.plot([], [], 'b-', lw=2)
point, = ax.plot([], [], 'ro')

def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

def update(frame):
    line.set_data(sigma[:frame], x[:frame])
    point.set_data(sigma[frame], x[frame])
    return line, point

ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=20)
plt.show()
