import pygame
import math
import numpy as np

# -------------------------------------
# Simulation / Display
# -------------------------------------
WIDTH, HEIGHT = 600, 600
FPS = 60
BASE_DT = 0.03  # base coordinate time step

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# We'll store all test particles here
# Each particle = [x, y, vx, vy, color]
particles = []

# -------------------------------------
# 1) Metric Components
# -------------------------------------
def g_tt(x, y):
    """
    Return g_{tt}(x,y).
    Example: - (1 + small potential)
    to mimic a Minkowski-like signature in time.
    If you want a purely positive approach, do e.g. +1.0.
    """
    ep = 10e-5
    # Example: Minkowski-like, no potential => g_tt = -1.0
    # Let's add a tiny "gravitational potential" effect:
    # e.g. g_tt = -(1.0 + 0.0002*y)
    r = math.sqrt((x-WIDTH/2)**2 + (y-HEIGHT/2)**2)
    val = -1.0 + 10.0/(r+ep)
    return val

def metric_spatial(x, y):
    """
    Return the 2x2 spatial block g_{ij}.
    We'll do a simple flat 2D for demonstration => diag(1,1).
    Replace with your own for curvature in x,y.
    """
    ep = 10e-5
    r = math.sqrt((x-WIDTH/2)**2 + (y-HEIGHT/2)**2)
    rs = 10.0
    g_xx = rs*x**2/((r**2+ep)*(rs-r+ep)) - 1
    g_xy = rs*x*y/((r**2+ep)*(rs-r+ep))
    g_yy = rs*y**2/((r**2+ep)*(rs-r+ep)) - 1

    return np.array([[g_xx, g_xy],
                     [g_xy, g_yy]], dtype=float)

def time_dilation_factor(x, y):
    """
    We'll define a local factor for dt => dt_local = dt / sqrt(|g_tt|).
    If g_tt is negative Minkowski-style => use -g_tt inside sqrt.
    """
    val = g_tt(x, y)
    if val < 0:
        # Minkowski-like: dt_local = dt / sqrt(-val)
        return math.sqrt(-val)
    else:
        # If user wants a purely positive g_tt, we do dt_local = dt / sqrt(val)
        return math.sqrt(val + 1e-10)

# -------------------------------------
# 2) Spatial Christoffels (2x2)
# -------------------------------------
EPS = 1e-4

def partial_derivative_g_spatial(x, y):
    """
    Finite difference partial derivatives of the 2x2 spatial block.
    """
    g0   = metric_spatial(x, y)
    gp_x = metric_spatial(x+EPS, y)
    gm_x = metric_spatial(x-EPS, y)
    gp_y = metric_spatial(x, y+EPS)
    gm_y = metric_spatial(x, y-EPS)

    dg_dx = (gp_x - gm_x)/(2*EPS)
    dg_dy = (gp_y - gm_y)/(2*EPS)
    return dg_dx, dg_dy

def christoffel_2d(x, y):
    """
    Compute Gamma^i_{jk} from the 2D metric g_{ij}.
    We'll store them in Gamma[i,j,k].
    """
    g_ij = metric_spatial(x, y)
    inv_g = np.linalg.inv(g_ij)

    dg_dx, dg_dy = partial_derivative_g_spatial(x, y)

    def partial_g(alpha, i, j):
        if alpha == 0:
            return dg_dx[i,j]
        else:
            return dg_dy[i,j]

    Gamma = np.zeros((2,2,2), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0.0
                for m in range(2):
                    val += inv_g[i,m]*(
                        partial_g(j, m, k)
                        + partial_g(k, m, j)
                        - partial_g(m, j, k)
                    )
                Gamma[i,j,k] = 0.5*val
    return Gamma

# -------------------------------------
# 3) Geodesic ODE in Spatial Coordinates
# -------------------------------------
def geodesic_accel(x, y, vx, vy):
    """
    d^2 x^i/dtau^2 = -Gamma^i_{jk} v^j v^k
    We'll treat v^0=vx, v^1=vy in the 2D space.
    """
    Gamma = christoffel_2d(x, y)
    # a_x = - sum_j,k Gamma[0,j,k]*v^j*v^k
    # a_y = - sum_j,k Gamma[1,j,k]*v^j*v^k
    ax = -(
        Gamma[0,0,0]*vx*vx +
        Gamma[0,0,1]*vx*vy +
        Gamma[0,1,0]*vy*vx +
        Gamma[0,1,1]*vy*vy
    )
    ay = -(
        Gamma[1,0,0]*vx*vx +
        Gamma[1,0,1]*vx*vy +
        Gamma[1,1,0]*vy*vx +
        Gamma[1,1,1]*vy*vy
    )
    return ax, ay

def update_particle(x, y, vx, vy):
    """
    We incorporate g_tt by adjusting the local proper-time step:
       dt_local = BASE_DT / sqrt(|g_tt(x,y)|).
    Then do an RK2 geodesic update in that local dt.
    """
    local_factor = time_dilation_factor(x, y)
    dt_local = BASE_DT / (local_factor if local_factor>1e-12 else 1e-12)

    # RK2
    ax1, ay1 = geodesic_accel(x, y, vx, vy)
    x_mid  = x + 0.5*dt_local*vx
    y_mid  = y + 0.5*dt_local*vy
    vx_mid = vx + 0.5*dt_local*ax1
    vy_mid = vy + 0.5*dt_local*ay1

    ax2, ay2 = geodesic_accel(x_mid, y_mid, vx_mid, vy_mid)

    x_new  = x + dt_local*vx_mid
    y_new  = y + dt_local*vy_mid
    vx_new = vx + dt_local*ax2
    vy_new = vy + dt_local*ay2

    return x_new, y_new, vx_new, vy_new

# -------------------------------------
# 4) Main Loop
# -------------------------------------
def main():
    running = True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    particles.clear()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # left-click => add a test particle
                    mx, my = pygame.mouse.get_pos()
                    # example initial velocity
                    vx, vy = (20.0, 0.0)
                    color = (255, 255, 255)  # white
                    particles.append([float(mx), float(my), vx, vy, color])

        # Update all particles
        new_list = []
        for (x, y, vx, vy, color) in particles:
            x_new, y_new, vx_new, vy_new = update_particle(x, y, vx, vy)
            # if the updated position is inside domain, keep it
            if 0 <= x_new <= WIDTH and 0 <= y_new <= HEIGHT:
                new_list.append([x_new, y_new, vx_new, vy_new, color])
            # else we discard the particle (it disappears)
        particles[:] = new_list

        # Draw
        screen.fill((0,0,0))  # black background

        # Draw particles as white circles
        for (x, y, vx, vy, color) in particles:
            pygame.draw.circle(screen, color, (int(x), int(y)), 3)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
