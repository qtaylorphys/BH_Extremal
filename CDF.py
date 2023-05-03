
import numpy as np
from mpmath import polylog, zeta
from scipy.optimize import root

from numbers import Real


@np.vectorize
def CDF(x: Real) -> Real:    
    cdf = (
        x**2 * np.log(1 - np.exp(-x))
        - 2 * (x * polylog(2, np.exp(-x)) + (polylog(3, np.exp(-x)) - zeta(3)))
    ) / (2 * zeta(3))

    return float(cdf)

@np.vectorize
def invert_CDF(x: Real, val: Real) -> Real:
    return CDF(x) - val

@np.vectorize
def find_x_from_CDF(val: Real) -> Real:
    x0 = val

    res = root(
        invert_CDF,
        x0,
        args=(val),
        method='lm',
        jac=None,
        tol=1e-17,
        callback=None,
        options=None,
    )

    return res.x[0]

if __name__ == "__main__":
    x_vals = np.linspace(1e-16, 23, 10000000)
    CDF_vals = CDF(x_vals)

    print(np.min(np.diff(CDF_vals)))
    print(CDF_vals[-1])

    CDF_data = np.stack((CDF_vals, x_vals), axis=-1)
    np.savetxt("CDF_data_new.csv.gz", CDF_data)
