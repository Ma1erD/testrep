import numpy as np

def erf(x, terms=50):
    """Compute the error function using numerical integration (Trapezoidal Rule)."""
    dx = x / terms
    total = 0.0
    for i in range(terms):
        xi = i * dx
        total += np.exp(-xi**2) + np.exp(-(xi+dx)**2)
    return (2 / np.sqrt(np.pi)) * (dx / 2) * total

def f(x, alpha=0.138e-6, t=60*24*60*60):
    """Function for root finding."""
    return erf(x / (2 * np.sqrt(alpha * t))) - 0.42857

def f_prime(x, alpha=0.138e-6, t=60*24*60*60):
    """Derivative of f(x)."""
    h = 1e-8  # Small step for numerical differentiation
    return (f(x + h, alpha, t) - f(x - h, alpha, t)) / (2 * h)

def bisection_method(a, b, tol=1e-13):
    """Bisection method implementation."""
    if f(a) * f(b) > 0:
        raise ValueError("Root is not bracketed.")
    while abs(b - a) > tol:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

def newtons_method(x0, tol=1e-13, max_iter=100):
    """Newton's method implementation."""
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        if abs(fpx) < 1e-14:  # Avoid division by very small number
            raise ValueError("Derivative too small.")
        x_new = x - fx / fpx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise ValueError("Newton's method did not converge.")

# Running the methods
x_bisection = bisection_method(0, 1)  # Start with interval [0,1]
x_newton = newtons_method(0.01)

# Print the results
print("Bisection Method Result:", x_bisection)
print("Newton's Method Result:", x_newton)

################################################################

import numpy as np
import matplotlib.pyplot as plt

# Function and its derivative
def f(x):
    return x**6 - x - 1

def df(x):
    return 6*x**5 - 1

# Newton's method
def newton_method(x0, tol=1e-10, max_iter=50):
    errors = []
    for _ in range(max_iter):
        x1 = x0 - f(x0) / df(x0)
        errors.append(abs(x1 - alpha))
        if abs(x1 - x0) < tol:
            break
        x0 = x1
    return errors

# Secant method
def secant_method(x0, x1, tol=1e-10, max_iter=50):
    errors = []
    for _ in range(max_iter):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        errors.append(abs(x2 - alpha))
        if abs(x2 - x1) < tol:
            break
        x0, x1 = x1, x2
    return errors

# Finding the largest root using a numerical solver for reference
from scipy.optimize import fsolve
alpha = fsolve(f, 2)[0]

# Running the methods
newton_errors = newton_method(2)
secant_errors = secant_method(2, 1)

# Log-log plot
plt.figure(figsize=(8, 6))
plt.loglog(newton_errors[:-1], newton_errors[1:], 'o-', label="Newton's Method")
plt.loglog(secant_errors[:-1], secant_errors[1:], 's-', label="Secant Method")
plt.xlabel(r'$|x_k - \alpha|$')
plt.ylabel(r'$|x_{k+1} - \alpha|$')
plt.title("Error Convergence in Log-Log Scale")
plt.show()
