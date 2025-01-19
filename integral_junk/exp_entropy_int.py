import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def funky(x):
    return (x-0.5)**4

def integrand(x, a):
    """
    Defines the integrand f(x, a) = x^x * (1 - x)^(1 - x) * e^(a * x).

    Parameters:
    - x: Integration variable (0 <= x <= 1)
    - a: Parameter

    Returns:
    - Value of the integrand at x
    """
    # Handle x=0 and x=1 by using the limits
    if x == 0.0:
        return 1.0 * (1 - x)**(x - 1) * np.exp(a * funky(x))  # x^x -> 1
    elif x == 1.0:
        return x**(-x) * 1.0 * np.exp(a * funky(x))  # (1 - x)^(1 -x) -> 1
    else:
        return (x**(-x)) * ((1 - x)**(x - 1)) * np.exp(a * funky(x))

def compute_Ia(a):
    """
    Computes the integral I(a) = ∫₀¹ x^x (1 - x)^(1 - x) e^(a x) dx

    Parameters:
    - a: Parameter

    Returns:
    - Numerical value of the integral I(a)
    """
    # Perform the integration using scipy.integrate.quad
    result, error = quad(integrand, 0, 1, args=(a,))
    return result

def normalized_integrand(x, a, Ia):
    """
    Computes the normalized integrand f(x, a) / I(a).

    Parameters:
    - x: Array of x values
    - a: Parameter
    - Ia: Integral I(a)

    Returns:
    - Normalized integrand values
    """
    # To avoid division by zero
    if Ia == 0:
        return np.zeros_like(x)
    return (x**(-x)) * ((1 - x)**(x - 1)) * np.exp(a * funky(x)) / Ia

def main():
    """
    Main function to compute and plot normalized integrands for different a values.
    """
    # Define the range of x values
    x_values = np.linspace(0, 1, 1000)

    # Define the list of 'a' values to plot
    a_list = [0, 2, 10, 20]  # You can change these values as desired

    # Initialize a plot
    plt.figure(figsize=(10, 6))

    for a in a_list:
        # Compute I(a)
        Ia = compute_Ia(a)

        # Compute normalized f(x,a)/I(a)
        y_values = normalized_integrand(x_values, a, Ia)

        # Plot the normalized integrand
        plt.plot(x_values, y_values, label=f'a = {a}')

    # Customize the plot
    plt.title(r'Normalized Integrand $f(x, a) / I(a)$ for Different Values of $a$')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x, a) / I(a)$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
