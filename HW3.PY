import numpy as np

def f(x):
    return 2*x-1-np.sin(x)

def bisection_method(f, a, b, tol=1e-6):
    if f(a) * f(b) >= 0:
        print(f"Bisection method fails: f({a}) * f({b}) >= 0")
        return None
    
    n = 0
    while (b - a) / 2 > tol and n <100:
        c = (a + b) / 2
        if f(c) == 0:  # Found exact root
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        n += 1
    
    return (a + b) / 2  # Approximate root

# Test cases
intervals = [(0,1)]

for a, b in intervals:
    root = bisection_method(f, a, b)
    print(f"Interval ({a}, {b}): Root found at {root}")

#question 2

def f(x):
    return (x - 5)**9

def bisection_method(f, a, b, tol=1e-4):
    if f(a) * f(b) >= 0:
        print(f"Bisection method fails: f({a}) * f({b}) >= 0")
        return None, 0
    
    n = 0
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if abs(f(c)) < tol:  # Adjusted stopping condition
            return c, n
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        n += 1
    
    return (a + b) / 2, n

# Define the interval
a, b = 4.82, 5.2

# Find the root using the bisection method
root, iterations = bisection_method(f, a, b)

# Print the results
print(f"Root found at x = {root:.8f} with {iterations} iterations.")

## Question 3

def f(x):
    return x**3 + x - 4

def bisection_method(f, a, b, tol=1e-3):
    if f(a) * f(b) >= 0:
        print(f"Bisection method fails: f({a}) * f({b}) >= 0")
        return None, 0
    
    n = 0
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if abs(f(c)) < tol:  # Adjusted stopping condition
            return c, n
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        n += 1
    
    return (a + b) / 2, n

# Define the interval
a, b = 1, 4

# Find the root using the bisection method
root, iterations = bisection_method(f, a, b)

# Print the results
print(f"Root found at x = {root:.6f} with {iterations} iterations.")


import matplotlib.pyplot as plt

# Define the function f(x)
def f(x):
    return x - 4 * np.sin(2*x) - 3

# Define the fixed-point iteration function
def g(x):
    return -np.sin(2*x) + (5*x)/4 - 3/4

# Plot the function
x_vals = np.linspace(-5, 5, 1000)
y_vals = f(x_vals)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label=r'$f(x) = x - 4\sin(2x) - 3$', color='b')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()

# Fixed-Point Iteration function
def fixed_point_iteration(g, x0, tol=1e-10, max_iter=100):
    x = x0
    for i in range(max_iter):
        x_new = g(x)
        if abs(x_new - x) < tol:
            return x_new, i  # Return root and number of iterations
        x = x_new
    return None, max_iter  # No convergence

# Finding roots using different starting points
initial_guesses = [-3, -1, 1, 3]
roots = []

for x0 in initial_guesses:
    root, iterations = fixed_point_iteration(g, x0)
    if root is not None:
        print(f"Root found at x = {root:.10f} with {iterations} iterations starting from x0 = {x0}")
        roots.append(root)
    else:
        print(f"Did not converge for initial guess x0 = {x0}")

# Display all found roots
print("\nFinal roots found:", np.round(roots, 10))
