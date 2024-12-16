import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog
import os

import sys

default_dir = "/home/connor/Dropbox/data/physics/"

########################################
# Example 3D Vector Fields
########################################
def v1(x, y, z):
    # rotation around z-axis
    return -y, x, 0

def v2(x, y, z):
    # radial outward from origin
    r = np.sqrt(x**2 + y**2 + z**2)
    if r == 0:
        return 0,0,0
    return x/r, y/r, z/r

def v3(x, y, z):
    # radial outward from origin
    r = np.sqrt(x**2 + y**2 + z**2)
    if r == 0:
        return 0,0,0
    return -y/r, x/r, z/r

vector_fields_3d = [
    {"name": "v1", "func": v1, "color": "blue", "visible": True, "type": "analytic"},
    {"name": "v2", "func": v2, "color": "red", "visible": True, "type": "analytic"},
    {"name": "v3", "func": v3, "color": "green", "visible": True, "type": "analytic"}
]

########################################
# Lie Bracket for 3D
########################################
def lie_bracket_3d(X, Y, x, y, z, h=1e-5):
    # Compute [X,Y]
    Xx, Xy, Xz = X(x,y,z)
    Yx, Yy, Yz = Y(x,y,z)

    def partials(F):
        # returns dF/dx, dF/dy, dF/dz at (x,y,z)
        Fx, Fy, Fz = F(x,y,z)

        # x-partial
        Fx_p, Fy_p, Fz_p = F(x+h,y,z)
        Fx_m, Fy_m, Fz_m = F(x-h,y,z)
        dFx_dx = (Fx_p - Fx_m)/(2*h)
        dFy_dx = (Fy_p - Fy_m)/(2*h)
        dFz_dx = (Fz_p - Fz_m)/(2*h)

        # y-partial
        Fx_p, Fy_p, Fz_p = F(x,y+h,z)
        Fx_m, Fy_m, Fz_m = F(x,y-h,z)
        dFx_dy = (Fx_p - Fx_m)/(2*h)
        dFy_dy = (Fy_p - Fy_m)/(2*h)
        dFz_dy = (Fz_p - Fz_m)/(2*h)

        # z-partial
        Fx_p, Fy_p, Fz_p = F(x,y,z+h)
        Fx_m, Fy_m, Fz_m = F(x,y,z-h)
        dFx_dz = (Fx_p - Fx_m)/(2*h)
        dFy_dz = (Fy_p - Fy_m)/(2*h)
        dFz_dz = (Fz_p - Fz_m)/(2*h)

        return (dFx_dx, dFy_dx, dFz_dx,
                dFx_dy, dFy_dy, dFz_dy,
                dFx_dz, dFy_dz, dFz_dz)

    dY = partials(Y)
    dX = partials(X)

    # [X,Y]^i = X^j dY^i/dx^j - Y^j dX^i/dx^j
    Lx = (Xx*dY[0] + Xy*dY[3] + Xz*dY[6]) - (Yx*dX[0] + Yy*dX[3] + Yz*dX[6])
    Ly = (Xx*dY[1] + Xy*dY[4] + Xz*dY[7]) - (Yx*dX[1] + Yy*dX[4] + Yz*dX[7])
    Lz = (Xx*dY[2] + Xy*dY[5] + Xz*dY[8]) - (Yx*dX[2] + Yy*dX[5] + Yz*dX[8])

    return Lx, Ly, Lz

def make_lie_bracket_field(X_func, Y_func, field1_name, field2_name):
    def LB_func(x,y,z):
        return lie_bracket_3d(X_func, Y_func, x, y, z)
    new_name = f"LB[{field1_name},{field2_name}]"
    new_field = {"name": new_name, "func": LB_func, "color": "purple", "visible": True, "type":"lie_bracket"}
    return new_field

def add_lie_bracket_field():
    f1_name = lb_field1_entry.get()
    f2_name = lb_field2_entry.get()

    # Find the fields
    Xf = None
    Yf = None
    for vf in vector_fields_3d:
        if vf["name"] == f1_name:
            Xf = vf["func"]
        if vf["name"] == f2_name:
            Yf = vf["func"]

    if Xf is None or Yf is None:
        print("One or both fields not found.")
        return

    new_field = make_lie_bracket_field(Xf, Yf, f1_name, f2_name)
    vector_fields_3d.append(new_field)
    add_checkbox_for_field(new_field)
    plot_vector_fields_3d()

########################################
# Loading/ Saving 3D Fields
########################################
def nearest_neighbor_interpolator_3d(x_arr, y_arr, z_arr, U_arr, V_arr, W_arr):
    def interp_func(x, y, z):
        i = np.argmin(np.abs(x_arr - x))
        j = np.argmin(np.abs(y_arr - y))
        k = np.argmin(np.abs(z_arr - z))
        return U_arr[k, j, i], V_arr[k, j, i], W_arr[k, j, i]
    return interp_func

def load_fields_from_dir_3d():
    dir_name = filedialog.askdirectory(initialdir=default_dir, title="Select Directory to Load 3D Data")
    if not dir_name:
        return

    for file in os.listdir(dir_name):
        if file.endswith(".csv"):
            filepath = os.path.join(dir_name, file)
            with open(filepath, 'r') as f:
                header = f.readline().strip()
            if header != "x,y,z,U,V,W":
                print(f"File {file} does not have expected header 'x,y,z,U,V,W'. Skipping.")
                continue

            data = np.loadtxt(filepath, delimiter=",", skiprows=1)
            xs = data[:,0]
            ys = data[:,1]
            zs = data[:,2]

            x_unique = np.unique(xs)
            y_unique = np.unique(ys)
            z_unique = np.unique(zs)

            X_size = len(x_unique)
            Y_size = len(y_unique)
            Z_size = len(z_unique)

            if X_size*Y_size*Z_size != data.shape[0]:
                print(f"{file} does not form a regular grid. Skipping.")
                continue

            # Sort by (z,y,x)
            sort_idx = np.lexsort((xs, ys, zs))
            data_sorted = data[sort_idx]
            xs_sorted = data_sorted[:,0]
            ys_sorted = data_sorted[:,1]
            zs_sorted = data_sorted[:,2]
            Us_sorted = data_sorted[:,3]
            Vs_sorted = data_sorted[:,4]
            Ws_sorted = data_sorted[:,5]

            X_grid = xs_sorted.reshape(Z_size, Y_size, X_size)
            Y_grid = ys_sorted.reshape(Z_size, Y_size, X_size)
            Z_grid = zs_sorted.reshape(Z_size, Y_size, X_size)
            U_grid = Us_sorted.reshape(Z_size, Y_size, X_size)
            V_grid = Vs_sorted.reshape(Z_size, Y_size, X_size)
            W_grid = Ws_sorted.reshape(Z_size, Y_size, X_size)

            x_line = X_grid[0,0,:]
            y_line = Y_grid[0,:,0]
            z_line = Z_grid[:,0,0]

            func = nearest_neighbor_interpolator_3d(x_line, y_line, z_line, U_grid, V_grid, W_grid)

            base_name = os.path.splitext(file)[0]
            new_field = {"name": base_name, "func": func, "color": "green", "visible": True, "type": "loaded"}
            vector_fields_3d.append(new_field)
            add_checkbox_for_field(new_field)

    plot_vector_fields_3d()

def save_vector_fields_3d():
    dir_name = filedialog.askdirectory(initialdir=default_dir, title="Select Directory to Save 3D Data")
    if not dir_name:
        return

    grid_size = 5
    x_vals = np.linspace(-2, 2, grid_size)
    y_vals = np.linspace(-2, 2, grid_size)
    z_vals = np.linspace(-2, 2, grid_size)
    X_grid, Y_grid, Z_grid = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

    for vf in vector_fields_3d:
        if vf["visible"]:
            data = []
            for i in range(X_grid.shape[0]):
                for j in range(X_grid.shape[1]):
                    for k in range(X_grid.shape[2]):
                        ux, vy, wz = vf["func"](X_grid[i,j,k], Y_grid[i,j,k], Z_grid[i,j,k])
                        data.append([X_grid[i,j,k], Y_grid[i,j,k], Z_grid[i,j,k], ux, vy, wz])
            filepath = os.path.join(dir_name, f"{vf['name']}_field.csv")
            np.savetxt(filepath, data, delimiter=",", header="x,y,z,U,V,W", comments='')
    print("3D fields saved.")

########################################
# Visibility and Plotting
########################################
def set_visibility_3d(name, state):
    for vf in vector_fields_3d:
        if vf["name"] == name:
            vf["visible"] = state
    plot_vector_fields_3d()

def plot_vector_fields_3d():
    ax_3d.clear()
    ax_3d.set_title("3D Vector Fields")

    grid_size = 5
    x_vals = np.linspace(-2, 2, grid_size)
    y_vals = np.linspace(-2, 2, grid_size)
    z_vals = np.linspace(-2, 2, grid_size)
    X_grid, Y_grid, Z_grid = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

    any_visible = False
    for vf in vector_fields_3d:
        if vf["visible"]:
            U = np.zeros(X_grid.shape)
            V = np.zeros(Y_grid.shape)
            W = np.zeros(Z_grid.shape)
            for i in range(X_grid.shape[0]):
                for j in range(X_grid.shape[1]):
                    for k in range(X_grid.shape[2]):
                        ux, vy, wz = vf["func"](X_grid[i,j,k], Y_grid[i,j,k], Z_grid[i,j,k])
                        U[i,j,k] = ux
                        V[i,j,k] = vy
                        W[i,j,k] = wz
            ax_3d.quiver(X_grid, Y_grid, Z_grid, U, V, W, length=0.3, color=vf["color"], label=vf["name"])
            any_visible = True

    if any_visible:
        ax_3d.legend()

    canvas_3d.draw()

########################################
# GUI Setup
########################################

root_3d = tk.Tk()
root_3d.title("3D Vector Field Viewer")
root_3d.geometry("1000x800")

# Proper window closing
def on_closing():
    root_3d.quit()
    root_3d.destroy()

root_3d.protocol("WM_DELETE_WINDOW", on_closing)

fig_3d = plt.figure(figsize=(8,6))
ax_3d = fig_3d.add_subplot(111, projection='3d')

canvas_3d = FigureCanvasTkAgg(fig_3d, master=root_3d)
canvas_3d.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

control_frame_3d = tk.Frame(root_3d)
control_frame_3d.pack(side=tk.RIGHT, fill=tk.Y)

tk.Label(control_frame_3d, text="Show/Hide Fields:").pack(anchor='w')

def make_toggle_command(name, var):
    return lambda: set_visibility_3d(name, var.get())

def add_checkbox_for_field(vf):
    var = tk.BooleanVar(value=vf["visible"])
    cb = tk.Checkbutton(control_frame_3d, text=vf["name"], variable=var, command=make_toggle_command(vf["name"], var))
    cb.pack(anchor='w')

# Add checkboxes for initial fields
for vf in vector_fields_3d:
    add_checkbox_for_field(vf)

# Buttons to load, save, and lie bracket
tk.Button(control_frame_3d, text="Load Fields", command=load_fields_from_dir_3d).pack(anchor='w', pady=10)
tk.Button(control_frame_3d, text="Save Fields", command=save_vector_fields_3d).pack(anchor='w', pady=10)

lb_frame = tk.Frame(control_frame_3d)
lb_frame.pack(anchor='w', pady=10)
tk.Label(lb_frame, text="Lie bracket of:").grid(row=0,column=0)
lb_field1_entry = tk.Entry(lb_frame)
lb_field1_entry.grid(row=0,column=1)
tk.Label(lb_frame, text=",").grid(row=0,column=2)
lb_field2_entry = tk.Entry(lb_frame)
lb_field2_entry.grid(row=0,column=3)
tk.Button(lb_frame, text="Compute LB", command=add_lie_bracket_field).grid(row=0,column=4)

plot_vector_fields_3d()

root_3d.mainloop()

# After the mainloop finishes (window closed), we can exit the script if needed:
sys.exit(0)
