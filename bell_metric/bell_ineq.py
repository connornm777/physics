import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

def f(x, y, z):
    """
    Computes the function f(x, y, z) = (x² + y² + z² - 1)² - 4(x²y² + y²z² + z²x² - 2xyz)
    """
    return (x**2 + y**2 + z**2 - 1)**2 - 4*(x**2 * y**2 + y**2 * z**2 + z**2 * x**2 - 2 * x * y * z)

def plot_heatmap(z_values, x_range=(-1, 1), y_range=(-1, 1), resolution=500):
    """
    Plots heatmaps of the function f for different slices of z with boundary lines for z = 1 - |x - y|.

    Parameters:
    - z_values: list or array of z slices to plot
    - x_range: tuple indicating the range of x values
    - y_range: tuple indicating the range of y values
    - resolution: number of points along each axis
    """
    # Create grid for x and y
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    num_plots = len(z_values)
    cols = 2  # Number of columns in subplot grid
    rows = (num_plots + cols - 1) // cols  # Compute required rows

    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    axes = axes.flatten()  # Flatten in case of multiple rows

    for idx, z0 in enumerate(z_values):
        ax = axes[idx]
        Z = z0

        # Compute f(x, y, z0)
        F = f(X, Y, Z)

        # Determine symmetric range for color scaling to enhance contrast
        abs_max = np.max(np.abs(F))
        abs_max = np.ceil(abs_max * 10) / 10  # Round up to nearest tenth for symmetry

        # Define a diverging colormap with high contrast
        cmap = cm.seismic  # 'seismic', 'bwr', or 'coolwarm' are good options
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

        # Plot the heatmap with enhanced contrast
        heatmap = ax.imshow(F, extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                           origin='lower', cmap=cmap, aspect='auto', norm=norm)

        # Add a colorbar
        cbar = fig.colorbar(heatmap, ax=ax)
        cbar.set_label('f(x, y, z)')

        # Calculate boundary lines for z = 1 - |x - y|
        c = 1 - Z  # c = 1 - z0

        if c >= 0:
            # Define x values within the range
            x_vals = np.linspace(x_range[0], x_range[1], 1000)

            # Calculate corresponding y values for the boundary lines
            y1 = x_vals - c
            y2 = x_vals + c

            # Clip y values to y_range
            y1_clipped = np.clip(y1, y_range[0], y_range[1])
            y2_clipped = np.clip(y2, y_range[0], y_range[1])

            # To avoid plotting lines outside the plot area
            valid1 = (y1 >= y_range[0]) & (y1 <= y_range[1])
            valid2 = (y2 >= y_range[0]) & (y2 <= y_range[1])

            ax.plot(x_vals[valid1], y1_clipped[valid1], color='yellow', linewidth=2, label='z = 1 - |x - y|')
            ax.plot(x_vals[valid2], y2_clipped[valid2], color='yellow', linewidth=2)

            # Optional: Add a legend for the boundary lines
            ax.legend(loc='upper right')

        # Set plot titles and labels
        ax.set_title(f'z = {z0}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # Set limits to ensure consistency across plots
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

    # If there are unused subplots, remove them
    for j in range(idx + 1, rows * cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define the z slices you want to plot within [-1, 1]
    z_values = [-0.9, -0.5, 0.0, 0.5, 0.9]

    plot_heatmap(z_values)
