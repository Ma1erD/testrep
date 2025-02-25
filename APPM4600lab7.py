import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
from scipy.special import comb

plt.rcParams['figure.figsize'] = [10, 5]

# Define the function
def f(x):
    return 1 / (1 + (10 * x)**2)

# Vandermonde interpolation routine
def poly_interp(f, xint, xtrg):
    n = len(xint) - 1
    fi = f(xint)
    V = np.vander(xint, increasing=True)
    c = np.linalg.solve(V, fi)
    g = np.polyval(c[::-1], xtrg)
    return g

# Lagrange interpolation
def lagrange_interp(f, xint, xtrg):
    lag_poly = lagrange(xint, f(xint))
    return lag_poly(xtrg)

# Newton interpolation
def newton_interp(f, xint, xtrg, srt=False):
    n = len(xint) - 1
    if srt:
        xint = sortxi(xint)
    fi = f(xint)
    D = np.zeros((n + 1, n + 1))
    D[:, 0] = fi
    for i in range(1, n + 1):
        D[i:n + 1, i] = (D[i:n + 1, i - 1] - D[i - 1:n, i - 1]) / (xint[i:n + 1] - xint[0:n + 1 - i])
    cN = D[0, :]
    g = cN[n] * np.ones_like(xtrg)
    for i in range(n - 1, -1, -1):
        g = g * (xtrg - xint[i]) + cN[i]
    return g

# Helper function to sort interpolation points
def sortxi(xi):
    n = len(xi) - 1
    a, b = xi[0], xi[n]
    xi2 = np.zeros(n + 1)
    xi2[0], xi2[1] = a, b
    xi = np.delete(xi, [0, n])
    for j in range(2, n + 1):
        dj = np.abs(xi2[j - 1] - xi)
        mj = np.argmax(dj)
        xi2[j] = xi[mj]
        xi = np.delete(xi, [mj])
    return xi2

# Updated Driver function for Exercise 3
def driver():
    xtrg = np.linspace(-1, 1, 1000)
    nn = np.arange(2, 21)  # N from 2 to 20

    plt.figure(figsize=(12, 6))

    for n in nn:
        xi = np.linspace(-1, 1, n + 1)

        # Monomial Expansion
        monomial_result = poly_interp(f, xi, xtrg)

        # Lagrange Interpolation
        lagrange_result = lagrange_interp(f, xi, xtrg)

        # Newton Interpolation
        newton_result = newton_interp(f, xi, xtrg)

        # Plot All Interpolations in the Same Plot
        plt.plot(xtrg, f(xtrg), 'k', label='Original Function' if n == 2 else "")
        plt.plot(xtrg, monomial_result, '--r', label=f'Monomial N={n}')
        plt.plot(xtrg, lagrange_result, '--g', label=f'Lagrange N={n}')
        plt.plot(xtrg, newton_result, '--b', label=f'Newton N={n}')
        plt.scatter(xi, f(xi), c='k', marker='o', label=f'Points N={n}')

        # Check for instability
        max_val = max(np.max(np.abs(monomial_result)), np.max(np.abs(lagrange_result)), np.max(np.abs(newton_result)))
        print(f'Max value of interpolation at N={n}: {max_val}')
        if max_val > 100:
            print(f"Interpolation becomes unstable at N = {n}")
            break

    plt.title('All Interpolations Combined')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Run the driver function
driver()
