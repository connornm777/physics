import pygame
import math

# -------------------------------------
# Domain & Basic Simulation Params
# -------------------------------------
WIDTH, HEIGHT = 400, 400
DT = 0.5

# "Infinite Summation" partial size
N = 0  # sum images in range i,j in [-N..N]

# Anti-periodic in x => crossing x flips sign of gamma
# Periodic in y => crossing y just wraps around
MOBIUS_X = False

# Vortex
VORTEX_RADIUS_CLAMP = 1.0
VORTEX_STRENGTH_DEFAULT = 1000.0

# Passive markers
# We'll have markers that do not affect the flow

# Velocity field shading
GRID_RES = 80
nfac=1.0
aaa=0.000
VEL_COLOR_CLAMP = 100.0

# Pygame init
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

print("Keyboard:")
print("  - 'v': create +vortex, shift+'v': create -vortex (at mouse)")
print("  - Left-click: place a marker")
print("  - 'r': reset")
print("  - ESC: quit")
print(f"Partial sum N={N}, MOBIUS_X={MOBIUS_X}, anti-periodic in x, periodic in y.")

# Data: Vortices: (x, y, gamma), Markers: (x, y)
vortices = []
markers = []

# -------------------------------------------------
# 1. Partial Summation of Images
# -------------------------------------------------
def partial_infinite_positions(xv, yv, width, height, n, mobius_x):
    """
    For each i in [-n..n], j in [-n..n], we produce an image (x_img, y_img, sign_factor).
    If mobius_x=True and i != 0:
      => y_img = height - y_img
      => If |i| is odd => sign_factor = -1
         else => +1
    For j != 0 in y => normal shift, no sign flip from crossing y.
    """
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            # Start with base offsets
            x_img = xv + i*width
            y_img = yv + j*height
            sign_factor = 1.0

            # If crossing x => flip y, possibly flip gamma
            if i != 0 and mobius_x:
                y_img = (height - y_img)
                # If the number of times we cross x is odd => gamma flips
                if abs(i) % 2 == 1:
                    sign_factor *= -1.0
    
            # For y, standard periodic => no sign flip
            yield (x_img, y_img, sign_factor)

def vortex_velocity_at(x, y, xv, yv, gamma):
    """
    Single vortex velocity: 2D point vortex with clamp at small r.
    """
    dx = x - xv
    dy = y - yv
    r2 = dx*dx + dy*dy
    if r2 < VORTEX_RADIUS_CLAMP*VORTEX_RADIUS_CLAMP:
        r2 = VORTEX_RADIUS_CLAMP*VORTEX_RADIUS_CLAMP
    fac = gamma / (2.0*math.exp(aaa*r2**(0.5))*math.pi*500*(r2/500)**nfac)
    ux = -dy * fac
    uy =  dx * fac
    return (ux, uy)

def total_velocity_at(x, y, vortices):
    """
    Sum velocity from each vortex's partial infinite images (2D).
    """
    rc = (x-WIDTH/2.0)**2+(y-HEIGHT/2.0)**2
    u_tot = 5000.0*(y-HEIGHT/2.0)/(2*math.pi*rc)
    v_tot = 5000.0*(WIDTH/2.0-x)/(2*math.pi*rc)
    for (xv, yv, g) in vortices:
        for (x_img, y_img, sign_factor) in partial_infinite_positions(xv, yv, WIDTH, HEIGHT, N, MOBIUS_X):
            # final gamma = g * sign_factor
            gamma_img = g * sign_factor
            ux, uy = vortex_velocity_at(x, y, x_img, y_img, gamma_img)
            u_tot += ux 
            v_tot += uy
    return (u_tot, v_tot)

# -------------------------------------------------
# 2. Domain Wrapping for Real Vortex
# -------------------------------------------------
def apply_boundary_vortex(x, y, gamma):
    """
    If x <0 or >= WIDTH => x+=±WIDTH; y->height-y; gamma->-gamma
    If y <0 or >= HEIGHT => y+=±HEIGHT (standard periodic) no sign flip.
    """
    # X checks
    if x < 0:
        x += WIDTH
        if MOBIUS_X:
            y = (HEIGHT - y)
            gamma = -gamma
    elif x >= WIDTH:
        x -= WIDTH
        if MOBIUS_X:
            y = (HEIGHT - y)
            gamma = -gamma

    # Y checks => standard periodic
    if y < 0:
        y += HEIGHT
    elif y >= HEIGHT:
        y -= HEIGHT

    return (x, y, gamma)

def apply_boundary_marker(x, y):
    """
    Markers are passive, but we still apply anti-periodic boundary if x crosses:
    => x±WIDTH, y->HEIGHT-y
    => y => standard periodic
    """
    if x < 0:
        x += WIDTH
        if MOBIUS_X:
            y = HEIGHT - y
    elif x >= WIDTH:
        x -= WIDTH
        if MOBIUS_X:
            y = HEIGHT - y

    if y < 0:
        y += HEIGHT
    elif y >= HEIGHT:
        y -= HEIGHT

    return (x, y)

# -------------------------------------------------
# Main Loop
# -------------------------------------------------
def main():
    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    vortices.clear()
                    markers.clear()
                elif event.key == pygame.K_v:
                    mx, my = pygame.mouse.get_pos()
                    mods = pygame.key.get_mods()
                    if (mods & pygame.KMOD_SHIFT) != 0:
                        gamma = -VORTEX_STRENGTH_DEFAULT
                    else:
                        gamma = +VORTEX_STRENGTH_DEFAULT
                    vortices.append((float(mx), float(my), gamma))

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # left-click => place a marker
                if event.button == 1:
                    mx, my = pygame.mouse.get_pos()
                    markers.append((float(mx), float(my)))

        # 1) Update vortices
        new_vorts = []
        for (xv, yv, g) in vortices:
            # sum velocity from images
            u, v = total_velocity_at(xv, yv, vortices)
            x_new = xv + u*DT
            y_new = yv + v*DT
            # apply domain wrap
            (x_bc, y_bc, g_bc) = apply_boundary_vortex(x_new, y_new, g)
            new_vorts.append((x_bc, y_bc, g_bc))
        vortices[:] = new_vorts

        # 2) Update markers
        new_marks = []
        for (xm, ym) in markers:
            u, v = total_velocity_at(xm, ym, vortices)
            x_new = xm + u*DT
            y_new = ym + v*DT
            (x_bc, y_bc) = apply_boundary_marker(x_new, y_new)
            new_marks.append((x_bc, y_bc))
        markers[:] = new_marks

        # 3) Render velocity field
        surf_small = pygame.Surface((GRID_RES, GRID_RES))
        dx = WIDTH / GRID_RES
        dy = HEIGHT / GRID_RES
        for gy in range(GRID_RES):
            y_samp = (gy + 0.5)*dy
            for gx in range(GRID_RES):
                x_samp = (gx + 0.5)*dx
                uu, vv = total_velocity_at(x_samp, y_samp, vortices)
                speed = math.sqrt(uu*uu + vv*vv)
                val = max(0, min(255, int(255*(speed/VEL_COLOR_CLAMP))))
                surf_small.set_at((gx, gy), (val, val, val))
        big_surf = pygame.transform.scale(surf_small, (WIDTH, HEIGHT))
        screen.blit(big_surf, (0,0))

        # 4) Draw vortices
        for (xv, yv, g) in vortices:
            color = (255,0,0) if g>0 else (0,0,255)
            pygame.draw.circle(screen, color, (int(xv), int(yv)), 4)

        # 5) Draw markers
        for (xm, ym) in markers:
            pygame.draw.circle(screen, (255,255,0), (int(xm), int(ym)), 2)

        pygame.display.flip()

    pygame.quit()

if __name__=="__main__":
    main()
