#!/usr/bin/env python3

import os
import json
import pickle
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path

import sympy as sp

BASE_PATH = Path("/home/connor/Dropbox/data/physics/symbolic_computation")

def create_session_folder():
    """
    Creates a new session folder under BASE_PATH with a timestamp-based name.
    Returns the Path object for the newly created session directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = BASE_PATH / f"session_{timestamp}"
    session_folder.mkdir(parents=True, exist_ok=True)
    return session_folder

def save_json(data, path):
    """
    Saves a dictionary to a JSON file.
    """
    with path.open("w") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    """
    Loads a dictionary from a JSON file.
    """
    with path.open("r") as f:
        return json.load(f)

def save_pickle(data, path):
    """
    Saves arbitrary Python objects to a pickle file.
    """
    with path.open("wb") as f:
        pickle.dump(data, f)

def load_pickle(path):
    """
    Loads a Python object from a pickle file.
    """
    with path.open("rb") as f:
        return pickle.load(f)

# -------------------------------------------------------------------------
# GUI PART 1: Collect Coordinates
# -------------------------------------------------------------------------
def build_gui_coordinates(session_path):
    """
    Builds a Tkinter GUI to collect coordinate names and proceed to metric entry.
    """
    root = tk.Tk()
    root.title("Symbolic Computation - Coordinate Entry")

    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    label = ttk.Label(frame, text="Enter coordinate names (comma-separated):")
    label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

    coord_entry = ttk.Entry(frame, width=40)
    coord_entry.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

    def submit_coordinates():
        """
        Parse user input (comma-separated coordinates) and save to coordinates.json.
        """
        raw_text = coord_entry.get().strip()
        if raw_text:
            coords = [c.strip() for c in raw_text.split(",") if c.strip()]
            if len(coords) == 0:
                messagebox.showerror("Input Error", "Please enter at least one coordinate.")
                return
            save_json({"coords": coords}, session_path / "coordinates.json")
            root.destroy()
            build_gui_metric(session_path)
        else:
            messagebox.showerror("Input Error", "Coordinate names cannot be empty.")

    submit_button = ttk.Button(frame, text="Submit", command=submit_coordinates)
    submit_button.grid(row=2, column=0, padx=5, pady=10, sticky=tk.E)

    return root

# -------------------------------------------------------------------------
# GUI PART 2: Collect Metric Components
# -------------------------------------------------------------------------
def build_gui_metric(session_path):
    """
    Builds a Tkinter GUI to collect metric tensor components g_{mu, nu}.
    """
    coords = load_json(session_path / "coordinates.json")["coords"]
    N = len(coords)

    # For a symmetric NxN metric, there are N(N+1)/2 unique components
    independent_components = [(i, j) for i in range(N) for j in range(i, N)]

    root = tk.Tk()
    root.title("Symbolic Computation - Metric Tensor Entry")

    canvas = tk.Canvas(root, borderwidth=0)
    frame = ttk.Frame(canvas, padding=10)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame.bind("<Configure>", on_frame_configure)

    instr_label = ttk.Label(
        frame,
        text="Enter metric tensor components g_{μν}:",
        font=("Arial", 12, "bold")
    )
    instr_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)

    entries = {}
    for idx, (i, j) in enumerate(independent_components, start=1):
        label_text = f"g_{{{coords[i]},{coords[j]}}} ="
        label = ttk.Label(frame, text=label_text)
        label.grid(row=idx, column=0, padx=5, pady=2, sticky=tk.E)

        entry = ttk.Entry(frame, width=40)
        entry.grid(row=idx, column=1, padx=5, pady=2, sticky=tk.W)
        entries[(i, j)] = entry

    def submit_metric():
        """
        Collect metric components from user input, validate, and save to metric.json.
        """
        metric = {}
        for (i, j), entry in entries.items():
            value = entry.get().strip()
            if not value:
                messagebox.showerror(
                    "Input Error",
                    f"Metric component g_{{{coords[i]},{coords[j]}}} cannot be empty."
                )
                return
            # Example key: "g_t,r"
            metric_key = f"g_{coords[i]},{coords[j]}"
            metric[metric_key] = value

        save_json({"metric": metric}, session_path / "metric.json")
        root.destroy()
        build_gui_summary(session_path)

    submit_button = ttk.Button(frame, text="Submit", command=submit_metric)
    submit_button.grid(
        row=len(independent_components) + 1,
        column=1,
        padx=5,
        pady=10,
        sticky=tk.E
    )

    return root

# -------------------------------------------------------------------------
# GUI PART 3: Summary GUI (Coordinates + Metric)
# -------------------------------------------------------------------------
def build_gui_summary(session_path):
    """
    Displays the summary of coordinates and metric components, then proceeds to SymPy.
    """
    coords = load_json(session_path / "coordinates.json")["coords"]
    metric = load_json(session_path / "metric.json")["metric"]

    root = tk.Tk()
    root.title("Symbolic Computation - Summary")

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill="both", expand=True)

    # Coordinates
    coord_label = ttk.Label(frame, text="Coordinates:", font=("Arial", 12, "bold"))
    coord_label.grid(row=0, column=0, sticky=tk.W)
    coords_text = ", ".join(coords)
    coord_value = ttk.Label(frame, text=coords_text, font=("Arial", 10))
    coord_value.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))

    # Metric
    metric_label = ttk.Label(frame, text="Metric Tensor Components:", font=("Arial", 12, "bold"))
    metric_label.grid(row=2, column=0, sticky=tk.W)
    metric_display = scrolledtext.ScrolledText(frame, width=60, height=10, wrap=tk.WORD, font=("Arial", 10))
    metric_display.grid(row=3, column=0, pady=(0, 10))

    for key, expr in metric.items():
        metric_display.insert(tk.END, f"{key} = {expr}\n")
    metric_display.configure(state='disabled')  # read-only

    def proceed():
        root.destroy()
        initiate_symbolic_computations(session_path)

    proceed_button = ttk.Button(frame, text="Proceed to Computations", command=proceed)
    proceed_button.grid(row=4, column=0, padx=5, pady=10, sticky=tk.E)

    return root

# -------------------------------------------------------------------------
# Symbolic Computation & Caching
# -------------------------------------------------------------------------
def parse_metric_to_sympy(session_path):
    """
    Parse the metric.json into a SymPy Matrix object, caching the result in metric_sympy.pkl.
    """
    pickle_path = session_path / "metric_sympy.pkl"
    if pickle_path.exists():
        # Load cached metric matrix
        return load_pickle(pickle_path)

    # Build metric from user input
    coords_data = load_json(session_path / "coordinates.json")
    coords = coords_data["coords"]
    metric_data = load_json(session_path / "metric.json")["metric"]

    sympy_coords = sp.symbols(coords)  # e.g. (t, r, theta, phi, ...)
    N = len(coords)

    metric_matrix = sp.zeros(N, N)
    for key, expr_str in metric_data.items():
        # Key is like g_0,1 => we parse i=0, j=1
        # (Because we stored the index references in the string)
        key_content = key.split("_")[1]  # e.g., "0,1"
        i_str, j_str = key_content.split(",")

        i = int(i_str)
        j = int(j_str)

        # Sympify the expression
        try:
            expr_sympy = sp.sympify(expr_str, locals={c: sympy_coords[idx] for idx, c in enumerate(coords)})
        except sp.SympifyError as e:
            raise ValueError(f"Error parsing expression for {key}: {e}")

        # Fill metric
        metric_matrix[i, j] = expr_sympy
        if i != j:
            metric_matrix[j, i] = expr_sympy  # enforce symmetry

    save_pickle(metric_matrix, pickle_path)
    return metric_matrix

def compute_and_cache(session_path, filename, compute_func):
    """
    Universal caching wrapper for any symbolic computation.
    """
    path = session_path / filename
    if path.exists():
        return load_pickle(path)
    else:
        result = compute_func()
        save_pickle(result, path)
        return result

def christoffel_symbols(g_inv, metric_matrix, coords):
    """
    Compute Christoffel symbols (Gamma^k_{i j}).
    Γ^k_{i j} = 1/2 * g^{k l} (∂g_{l j}/∂x^i + ∂g_{i l}/∂x^j - ∂g_{i j}/∂x^l)
    Returns a 3D array/list: Gamma[k, i, j].
    """
    N = len(coords)
    Gamma = [[[0]*N for _ in range(N)] for __ in range(N)]
    for k in range(N):
        for i in range(N):
            for j in range(N):
                cs_expr = sp.Rational(0)
                for l in range(N):
                    cs_expr += g_inv[k, l] * (
                        metric_matrix.diff(coords[i], l, j) +
                        metric_matrix.diff(coords[j], l, i) -
                        metric_matrix.diff(l, i, j)
                    ) / 2  # factor of 1/2
                Gamma[k][i][j] = sp.simplify(cs_expr)
    return Gamma

def compute_riemann_christoffel(Gamma, g_inv, metric_matrix, coords):
    """
    Compute the Riemann curvature tensor R^rho_{sigma mu nu}
    using Christoffel symbols. We then store as R[rho, sigma, mu, nu].
    """
    N = len(coords)
    R = [[[[0]*N for _ in range(N)] for __ in range(N)] for ___ in range(N)]
    # R^rho_{sigma mu nu} = ∂Γ^rho_{sigma nu}/∂x^mu - ∂Γ^rho_{sigma mu}/∂x^nu
    #                       + Γ^rho_{lambda mu} Γ^lambda_{sigma nu} - Γ^rho_{lambda nu} Γ^lambda_{sigma mu}
    for rho in range(N):
        for sigma in range(N):
            for mu in range(N):
                for nu in range(N):
                    term = Gamma[rho][sigma][nu].diff(coords[mu]) - Gamma[rho][sigma][mu].diff(coords[nu])
                    for lam in range(N):
                        term += Gamma[rho][lam][mu]*Gamma[lam][sigma][nu]
                        term -= Gamma[rho][lam][nu]*Gamma[lam][sigma][mu]
                    R[rho][sigma][mu][nu] = sp.simplify(term)
    return R

def compute_ricci(R):
    """
    Ricci tensor R_{sigma nu} = R^rho_{sigma rho nu}
    Takes 4D array R[rho, sigma, mu, nu] and contracts on (rho, mu).
    """
    N = len(R)
    # R is 4D: R[rho][sigma][mu][nu]
    # Ricci is 2D: R[sigma, nu] = sum_{rho} R[rho][sigma][rho][nu]
    ricci = [[sp.Rational(0) for _ in range(N)] for __ in range(N)]
    for sigma in range(N):
        for nu in range(N):
            val = sp.Rational(0)
            for rho in range(N):
                val += R[rho][sigma][rho][nu]
            ricci[sigma][nu] = sp.simplify(val)
    return ricci

def compute_scalar_curvature(g_inv, ricci):
    """
    R = g^{mu nu} R_{mu nu}.
    """
    N = g_inv.shape[0]
    scalar = sp.Rational(0)
    for mu in range(N):
        for nu in range(N):
            scalar += g_inv[mu, nu] * ricci[mu][nu]
    return sp.simplify(scalar)

# -------------------------------------------------------------------------
# GUI PART 4: Initiate Symbolic Computations & Show Results
# -------------------------------------------------------------------------
def initiate_symbolic_computations(session_path):
    """
    Handles the full chain of symbolic computations: parse metric, compute:
      - determinant
      - inverse metric
      - Christoffel symbols
      - Riemann, Ricci, scalar curvature
    Displays a final results GUI.
    """
    root = tk.Tk()
    root.title("Symbolic Computation - Curvature Computations")

    # Load or create SymPy metric
    metric_matrix = parse_metric_to_sympy(session_path)

    # Coordinates as Sympy symbols
    coords_data = load_json(session_path / "coordinates.json")
    coord_syms = sp.symbols(coords_data["coords"])
    N = len(coord_syms)

    # DET & INVERSE
    def compute_determinant():
        return metric_matrix.det()

    def compute_inverse():
        return metric_matrix.inv()

    g_det = compute_and_cache(session_path, "metric_det.pkl", compute_determinant)
    g_inv = compute_and_cache(session_path, "metric_inv.pkl", compute_inverse)

    # Christoffel
    def compute_christoffel():
        return christoffel_symbols(g_inv, metric_matrix, sp.Matrix(coord_syms))
    Gamma = compute_and_cache(session_path, "Gamma.pkl", compute_christoffel)

    # Riemann
    def compute_riemann():
        return compute_riemann_christoffel(Gamma, g_inv, metric_matrix, sp.Matrix(coord_syms))
    Riemann = compute_and_cache(session_path, "Riemann.pkl", compute_riemann)

    # Ricci
    def compute_ricci_tensor():
        return compute_ricci(Riemann)
    ricci_tensor = compute_and_cache(session_path, "ricci.pkl", compute_ricci_tensor)

    # Scalar
    def compute_scalar():
        return compute_scalar_curvature(g_inv, ricci_tensor)
    scalar_curv = compute_and_cache(session_path, "scalar.pkl", compute_scalar)

    # ---------------------------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------------------------
    frame = ttk.Frame(root, padding=10)
    frame.pack(fill="both", expand=True)

    label = ttk.Label(frame, text="Final Symbolic Computations:", font=("Arial", 12, "bold"))
    label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

    results_box = scrolledtext.ScrolledText(frame, width=60, height=20, wrap=tk.WORD)
    results_box.grid(row=1, column=0, sticky=(tk.W, tk.E))

    results = []
    results.append(f"Det(g) = {sp.simplify(g_det)}\n")
    results.append(f"g^(-1) =\n{sp.simplify(g_inv)}\n\n")

    # Christoffel is large. Just show a single example or let user expand
    results.append("Christoffel Symbols (Gamma^k_{i j}): (showing a few)\n")
    try:
        # Show only Gamma^0_{0,0}, Gamma^0_{1,1} for brevity
        # in a real app, you might display them all or provide a scrolling pane
        results.append(f"Gamma^0_{{0,0}} = {Gamma[0][0][0]}\n")
        if N > 1:
            results.append(f"Gamma^0_{{1,1}} = {Gamma[0][1][1]}\n")
        results.append("...\n\n")
    except:
        pass

    # Riemann, Ricci, Scalar
    results.append("Ricci Tensor (R_{μν}):\n")
    for i in range(N):
        row_str = ", ".join(str(ricci_tensor[i][j]) for j in range(N))
        results.append(f"Row {i}: [{row_str}]\n")

    results.append(f"\nScalar Curvature R = {scalar_curv}\n")

    results_box.insert(tk.END, "".join(results))
    results_box.configure(state='disabled')

    def finish():
        root.destroy()
        print("[INFO] Computations completed. Session data stored in:")
        print(session_path)

    finish_button = ttk.Button(frame, text="Finish", command=finish)
    finish_button.grid(row=2, column=0, pady=10, sticky=tk.E)

    root.mainloop()

def main():
    """
    1. Create a new session folder
    2. Prompt for coordinates
    3. Prompt for metric components
    4. Compute geometry quantities
    """
    session_path = create_session_folder()
    print(f"[INFO] New session folder created at: {session_path}")

    # Collect coordinates
    root = build_gui_coordinates(session_path)
    root.mainloop()

if __name__ == "__main__":
    main()
