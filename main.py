import numpy as np
from mpmath import polylog, zeta
from scipy.optimize import root

from numbers import Real

import numba as nb

from interpolate import cubic_spline

from time import process_time

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

@nb.njit(
    fastmath = True,
)
def compute_BH_evolution(
    M_init,
    J_init,
    eps,
    dist,
):
    M = M_init
    J = J_init
    i = 0
    np.random.seed(1234)

    result = np.empty((100000000000, 3))

    while M >= 1.:
        a_star = J / M**2.

        result[i, :] = M, J, a_star
        if np.abs(a_star) > 1. - eps:
            a_star = 1.
            break

        rho = 1. / 2. * (1. - 2. * a_star)**2.

        T = np.sqrt(1 - a_star**2) / (8 * np.pi * M * (1 + np.sqrt(1 - a_star**2)))

        change = dist(np.random.rand())

        M = M - change * T

        if rho > np.random.rand():
            J = J - change * T
        else:
            J = J + change * T

        i += 1

        # if i > 99999: print("Problem")

    return result[:i+1, :]

@nb.njit(
    # fastmath = True,
)
def compute_BH_evolution_new(
    M_init,
    J_init,
    eps,
    dist,
):
    M = M_init
    J = J_init
    i = 0
    np.random.seed(1234)

    np.random.rand(100)

if __name__ == "__main__":
    cdffunction = np.genfromtxt("cdffunction.csv")
    y_points = np.linspace(0.000001, 12, 10001)

    # def dist(x):
    #     invfunction = interpolate.splrep(cdffunction, y_points)
    #     return interpolate.splev(x, invfunction)
    
    inv_CDF = cubic_spline(cdffunction, y_points)
    
    init_M = 2000.
    init_J = 0.
    eps = 1e-4

    result = compute_BH_evolution(init_M, init_J, eps, inv_CDF)
    compute_BH_evolution_new(init_M, init_J, eps, inv_CDF)

    print(result[-1])
    t1 = process_time()
    result = compute_BH_evolution(init_M, init_J, eps, inv_CDF)
    t2 = process_time()
    compute_BH_evolution_new(init_M, init_J, eps, inv_CDF)
    t3 = process_time()

    print(t2 - t1, t3 - t2)
