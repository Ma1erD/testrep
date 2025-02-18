import numpy as np
from scipy.linalg import lu_factor, lu_solve


def F(x):
    return np.array([4*x[0]**2 + x[1]**2 - 4,x[0] + x[1] - np.sin(x[0] - x[1])])

def JF(x):
    return np.array([[8*x[0], 2*x[1]],[1 - np.cos(x[0] - x[1]), 1 + np.cos(x[0] - x[1])]])

def slacker_newton(f, Jf, x0, tol=1e-10, nmax=100, threshold=1e-2):
    """ Slacker Newton: Recomputes Jacobian when convergence slows. """
    x = x0
    J = Jf(x)
    lu, piv = lu_factor(J)
    
    for _ in range(nmax):
        F = f(x)
        if np.linalg.norm(F) < tol:
            return x
        
        p = -lu_solve((lu, piv), F)
        x_new = x + p
        
        if np.linalg.norm(p) > threshold * np.linalg.norm(x):
            J = Jf(x_new)
            lu, piv = lu_factor(J)
        
        x = x_new
    
    return x


x0 = np.array([1.0, 0.0])
root = slacker_newton(F, JF, x0)
print("Root approximation:", root)
