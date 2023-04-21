import numpy as np
from mpmath import polylog, zeta
from scipy.optimize import root

def CDF(x):
    if isinstance(x, (list, np.ndarray)): x = x[0]
    
    cdf = (x**2 * np.log(1 - np.exp(-x)) - 2 * (x * polylog(2, np.exp(-x)) + (polylog(3, np.exp(-x)) - zeta(3)))) / (2 * zeta(3))
    return float(cdf)

def invert_CDF(x, val):
    return CDF(x) - val

def find_x_from_CDF(val):
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

