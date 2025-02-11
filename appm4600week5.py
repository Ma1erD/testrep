import numpy as np
import matplotlib.pyplot as plt

def fun(x):
    return np.exp(x**2 + 7*x - 30) - 1

def dfun(x):
    return (2*x + 7) * np.exp(x**2 + 7*x - 30)

def bisection_method(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) > 0:
        raise ValueError("Function must have opposite signs at endpoints a and b")
    
    iter_count = 0
    while (b - a) / 2 > tol and iter_count < max_iter:
        c = (a + b) / 2  # Midpoint
        if f(c) == 0:
            return c, iter_count
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iter_count += 1
        
    return (a + b) / 2, iter_count

def newton_method(f, df, x0, tol=1e-14, max_iter=100):
    xn = x0
    iter_count = 0
    while abs(f(xn)) > tol and iter_count < max_iter:
        dfx = df(xn)
        if abs(dfx) < 1e-10:
            raise ValueError("Derivative near zero, Newton method fails")
        xn -= f(xn) / dfx
        iter_count += 1
    return xn, iter_count

















def hybrid_method(f, df, a, b, tol=1e-6, max_iter=100):
    mid, bisect_iters = bisection_method(f, a, b, tol, max_iter)
    root, newton_iters = newton_method(f, df, mid, tol, max_iter)
    total_iters = bisect_iters + newton_iters
    return root, total_iters, bisect_iters, newton_iters

# Define the problem
a, b = 2, 4.5
x0 = 4.5

# Run methods
bisect_root, bisect_iters = bisection_method(fun, a, b)
newton_root, newton_iters = newton_method(fun, dfun, x0)
hybrid_root, total_iters, bisect_part, newton_part = hybrid_method(fun, dfun, a, b)

# Display results
print(f"Bisection Method: Root = {bisect_root:.6f}, Iterations = {bisect_iters}")
print(f"Newton's Method: Root = {newton_root:.6f}, Iterations = {newton_iters}")
print(f"Hybrid Method: Root = {hybrid_root:.6f}, Total Iterations = {total_iters}, Bisection Part = {bisect_part}, Newton Part = {newton_part}")
