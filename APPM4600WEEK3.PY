import numpy as np

def f(x):
    return x**2 * (x - 1)

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
#intervals = [(0.5, 2), (-1, 0.5), (-1, 2)]

#for a, b in intervals:
    root = bisection_method(f, a, b)
    print(f"Interval ({a}, {b}): Root found at {root}")



def f1(x):
    return (x - 1) * (x - 3) * (x - 5)

def f2(x):
    return (x - 1)**2 * (x - 3)

def f3(x):
    return np.sin(x)


# Test cases
intervals = [(f1, 0, 2.4), (f2, 0, 2), (f3, 0, 0.1), (f3, 0.5, np.pi/4)]

for func, a, b in intervals:
    root = bisection_method(func, a, b)
    print(f"Interval ({a}, {b}): Root found at {root}")

