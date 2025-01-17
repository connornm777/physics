import numpy as np
import matplotlib.pyplot as plt

def f(x, N):
    """
    Computes f(x, N) = sum_{n=-N to N} 2^n sin^2(pi/2 * floor(x * 2^(-n))).
    x can be a scalar or a NumPy array.
    """
    # Ensure x is an array for vectorized operations
    x_array = np.array(x, dtype=float)
    result = np.zeros_like(x_array)

    for n in range(-N, N + 1):
        # floor(...) is applied elementwise by np.floor
        # sin(...) is also elementwise
        argument = (np.pi / 2.0) * np.floor(x_array * 2.0 ** (-n))
        result += (2.0 ** n) * np.sin(argument) ** 2

    return result

def main():
    # Define a range of x values over which to plot
    x_vals = np.linspace(0, 10, 2001)  # 2001 points from 0 to 10

    # Plot for various values of N
    for N_val in [1, 2, 3, 4, 5]:
        y_vals = f(x_vals, N_val)
        plt.plot(x_vals, y_vals, label=f"N = {N_val}")

    plt.title(r"Plot of $f(x, N) = \sum_{n=-N}^{N} 2^n \sin^2\left(\frac{\pi}{2}\lfloor x 2^{-n}\rfloor\right)$")
    plt.xlabel("x")
    plt.ylabel("f(x, N)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
