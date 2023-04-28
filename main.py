import numpy as np
from mpmath import polylog, zeta
from scipy.optimize import root
from scipy import interpolate

from numbers import Real

import pandas as pd

def CDF(x: Real) -> Real:
    if isinstance(x, (list, np.ndarray)): x = x[0]
    
    cdf = (
        x**2 * np.log(1 - np.exp(-x))
        - 2 * (x * polylog(2, np.exp(-x)) + (polylog(3, np.exp(-x)) - zeta(3)))
    ) / (2 * zeta(3))
    return float(cdf)

def invert_CDF(x: Real, val: Real) -> Real:
    return CDF(x) - val

def find_x_from_CDF(val: Real) -> Real:
    x0 = 5e-1

    res = root(
        invert_CDF,
        x0,
        args=(val),
        method='lm',
        jac=None,
        tol=1e-9,
        callback=None,
        options=None,
    )

    return res.x[0]

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    cdffunction = pd.read_csv('cdffunction.csv', header = None)
    
    y_points = np.linspace(0.000001, 12, 10001)

    def dist(x):
        invfunction = interpolate.splrep(cdffunction, y_points)
        return interpolate.splev(x, invfunction)
    
    init_M = 100
    M = init_M

    init_J = 0
    J = init_J

    i = 0

    eps = 1e-4
    
    while M >= 1:
        print(i)

        a_star = J / M**2
        print(M, J, a_star)
        print()
        if np.abs(a_star) > 1 - eps:
            a_star = 1
            break

        rho = 1 / 2 * (1 - 2 * a_star)**2
        T = 1 / (8 * np.pi * M)

        probs = rng.random()
        rands = rng.random()

        change = dist(probs)

        M = M - change * T

        if rho > rands:
            J = J - change * T
        else:
            J = J + change * T

        i += 1
            

