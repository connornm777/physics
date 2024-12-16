import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import sys

# ==========================
# Configuration
# ==========================

grid_size = 30
x_vals = np.linspace(-2, 2, grid_size)
y_vals = np.linspace(-2, 2, grid_size)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

# ==========================
# Define Vector Fields
# ==========================

def e1(x, y):
    return np.cos(y), np.sin(y)

def e2(x, y):
    return -np.sin(y), np.cos(y)

def e3(x, y):
    return 1, 0

def e4(x, y):
    return 0, 1


# Initialize the fields: store arrays for U,V directly
def create_field(name, func, color):
    U = np.zeros((grid_size, grid_size))
    V = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            U[i, j], V[i, j] = func(X_grid[i,j], Y_grid[i,j])
    return {
        "name": name,
        "func": func,
        "U": U,
        "V": V,
        "color": color,
        "visible": True
    }

vector_fields = [
    create_field("e1", e1, "blue"),
    create_field("e2", e2, "red"),
    create_field("e3", e3, "green"),
    create_field("e4", e4, "purple")
]

# ==========================
# Lie Bracket Computation
# ==========================

def lie_bracket_2d(X_func, Y_func, x, y, h=1e-5):
    # Compute partials of Y
    Yx_plus, Yy_plus = Y_func(x+h, y)
    Yx_minus, Yy_minus = Y_func(x-h, y)
    dYx_dx = (Yx_plus - Yx_minus)/(2*h)
    dYy_dx = (Yy_plus - Yy_minus)/(2*h)

    Yx_plus, Yy_plus = Y_func(x, y+h)
    Yx_minus, Yy_minus = Y_func(x, y-h)
    dYx_dy = (Yx_plus - Yx_minus)/(2*h)
    dYy_dy = (Yy_plus - Yy_minus)/(2*h)

    # Compute partials of X
    Xx_plus, Xy_plus = X_func(x+h, y)
    Xx_minus, Xy_minus = X_func(x-h, y)
    dXx_dx = (Xx_plus - Xx_minus)/(2*h)
    dXy_dx = (Xy_plus - Xy_minus)/(2*h)

    Xx_plus, Xy_plus = X_func(x, y+h)
    Xx_minus, Xy_minus = X_func(x, y-h)
    dXx_dy = (Xx_plus - Xx_minus)/(2*h)
    dXy_dy = (Xy_plus - Xy_minus)/(2*h)

    Xu, Xv = X_func(x, y)
    Yu, Yv = Y_func(x, y)

    # [X,Y]^x = Xu*dYx/dx + Xv*dYx/dy - (Yu*dXx/dx + Yv*dXx/dy)
    Lx = Xu*dYx_dx + Xv*dYx_dy - (Yu*dXx_dx + Yv*dXx_dy)
    # [X,Y]^y = Xu*dYy/dx + Xv*dYy/dy - (Yu*dXy/dx + Yv*dXy/dy)
    Ly = Xu*dYy_dx + Xv*dYy_dy - (Yu*dXy_dx + Yv*dXy_dy)

    return Lx, Ly

def compute_lie_bracket_field(field1_name, field2_name):
    # Find the fields
    f1 = None
    f2 = None
    for vf in vector_fields:
        if vf["name"] == field1_name:
            f1 = vf
        if vf["name"] == field2_name:
            f2 = vf

    if f1 is None or f2 is None:
        print("One or both fields not found.")
        return None

    # Compute LB arrays
    Lx_arr = np.zeros((grid_size, grid_size))
    Ly_arr = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            x, y = X_grid[i,j], Y_grid[i,j]
            Lx_arr[i,j], Ly_arr[i,j] = lie_bracket_2d(f1["func"], f2["func"], x, y)

    lb_name = f"LB[{field1_name},{field2_name}]"
    return {
        "name": lb_name,
        "func": None,  # Not needed since we have arrays directly
        "U": Lx_arr,
        "V": Ly_arr,
        "color": "purple",
        "visible": True
    }

# ==========================
# Plotting
# ==========================

def plot_vector_fields():
    ax.clear()
    ax.set_title("2D Vector Fields with Lie Bracket")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    for vf in vector_fields:
        if vf["visible"]:
            ax.quiver(X_grid, Y_grid, vf["U"], vf["V"], color=vf["color"], label=vf["name"], pivot='mid')
    ax.legend()
    canvas.draw()

# ==========================
# GUI Functions
# ==========================

def set_visibility(name, state):
    for vf in vector_fields:
        if vf["name"] == name:
            vf["visible"] = state
    plot_vector_fields()

def add_checkbox_for_field(vf):
    var = tk.BooleanVar(value=vf["visible"])
    cb = tk.Checkbutton(control_frame, text=vf["name"], variable=var,
                        command=lambda n=vf["name"], v=var: set_visibility(n, v.get()))
    cb.pack(anchor='w')

def compute_lb_button():
    f1 = lb_field1_entry.get().strip()
    f2 = lb_field2_entry.get().strip()
    new_field = compute_lie_bracket_field(f1, f2)
    if new_field is not None:
        vector_fields.append(new_field)
        add_checkbox_for_field(new_field)
        plot_vector_fields()

# ==========================
# Main GUI Setup
# ==========================

root = tk.Tk()
root.title("2D Vector Field Viewer - Minimal")
root.geometry("1000x700")

def on_closing():
    root.quit()
    root.destroy()
    sys.exit(0)

root.protocol("WM_DELETE_WINDOW", on_closing)

fig, ax = plt.subplots(figsize=(6,5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

control_frame = tk.Frame(root)
control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

tk.Label(control_frame, text="Show/Hide Fields:", font=("Helvetica", 12, "bold")).pack(anchor='w')

for vf in vector_fields:
    add_checkbox_for_field(vf)

tk.Label(control_frame, text="Compute Lie Bracket:", font=("Helvetica", 12, "bold")).pack(anchor='w', pady=(20,0))
lb_frame = tk.Frame(control_frame)
lb_frame.pack(anchor='w', pady=5)

tk.Label(lb_frame, text="Field 1:").grid(row=0, column=0, padx=5, pady=2)
lb_field1_entry = tk.Entry(lb_frame, width=10)
lb_field1_entry.grid(row=0, column=1, padx=5, pady=2)

tk.Label(lb_frame, text="Field 2:").grid(row=1, column=0, padx=5, pady=2)
lb_field2_entry = tk.Entry(lb_frame, width=10)
lb_field2_entry.grid(row=1, column=1, padx=5, pady=2)

tk.Button(lb_frame, text="Compute LB", command=compute_lb_button).grid(row=2, column=0, columnspan=2, pady=5)

plot_vector_fields()

root.mainloop()
sys.exit()