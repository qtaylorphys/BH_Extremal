import argparse
import numpy as np
from mpmath import polylog, zeta, mp
from scipy.optimize import root
import scipy.interpolate as spip
from time import process_time
import h5py
import os

from dateutil.relativedelta import relativedelta as rd

from numbers import Real
from nptyping import NDArray


@np.vectorize
def CDF(x: Real) -> Real:
    """
    Compute the cumulative density function (CDF) of the black body 
    distribution; for derivation see Mathematica_files/CDF_derivation.nb

    Parameters:
    x (Real): argument of the CDF, equal to E/T

    Returns:
    Real: the value of the CDF
    """
    mp.dps = 20

    cdf = (
        x**2 * np.log(1 - np.exp(-x))
        - 2 * (x * polylog(2, np.exp(-x)) + (polylog(3, np.exp(-x)) - zeta(3)))
    ) / (2 * zeta(3))

    return float(cdf)

@np.vectorize
def invert_CDF(x: Real, val: Real) -> Real:
    """
    Function to aid root finding for the CDF, essentially the LHS of
        CDF(x) - val = 0

    Parameters:
    x (Real): argument of the CDF, equal to E/T
    val (Real): the value of the CDF to be inverted

    Returns:
    Real: the LHS of the above equation
    """
    return CDF(x) - val

@np.vectorize
def find_x_from_CDF(val: Real) -> Real:
    """
    Performs root finding for the CDF, namely solving
        CDF(x) = val
    for x

    Parameters:
    val (Real): the RHS of the above equation

    Returns:
    Real: the root of the above equation
    """
    x0 = val

    res = root(
        invert_CDF,
        x0,
        args=(val),
        method='lm',
        jac=None,
        tol=1e-12,
        callback=None,
        options=None,
    )

    return res.x[0]

def binary_size(num: Real, suffix: str = "B") -> str:
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"

def time_fmt(time_s: Real) -> str:
    intervals = ["days", "hours", "minutes", "seconds"]
    x = rd(seconds=time_s)
    vals = [getattr(x, k) for k in intervals]
    
    if not all(v == 0 for v in vals[:-1]):
        vals = [int(v) for v in vals]
    else:
        vals[-1] = np.round(vals[-1], 1)

    return ' '.join(
        f"{val} {interval}"
        for val, interval in zip(vals, intervals) if val != 0
    )

def save_data_h5(data: NDArray, filename: str, dataset_name: str) -> None:
    with h5py.File(filename, 'w') as f:
        f.create_dataset(
            dataset_name,
            data=data,
            compression="gzip",
            compression_opts=9,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute values of the CDF",
    )

    parser.add_argument("size", type=int)
    parser.add_argument("--interp_coeffs", action="store_true")
    args = parser.parse_args()

    N = int(args.size)

    x_min = 1e-16
    x_max = 23.75

    x_vals = np.linspace(x_min, x_max, N)
    print(f"Generated {N} equally spaced values between {x_min:.1e} and {x_max:.3e}")

    t_start = process_time()
    CDF_vals = CDF(x_vals)
    t_end = process_time()
    t_elapsed = time_fmt(t_end - t_start)
    print(f"Computed {N} values of the CDF in {t_elapsed}")

    if np.min(np.diff(CDF_vals)) <= 0:
        raise Exception("CDF values are not monotonically increasing.")
    
    CDF_data = np.stack((CDF_vals, x_vals), axis=-1)

    save_data_h5(CDF_data, "CDF_data.h5", "CDF_data")

    file_stats = os.stat("CDF_data.h5")
    file_size = binary_size(file_stats.st_size)
    print(f"Saved data in file CDF_data.h5 with size {file_size}")

    if args.interp_coeffs:
        inv_CDF_interp = spip.CubicSpline(CDF_vals, x_vals)
        interp_coeffs = inv_CDF_interp.c

        save_data_h5(interp_coeffs.T, "CDF_coeffs.h5", "CDF_coeffs")

        file_stats = os.stat("CDF_coeffs.h5")
        file_size = binary_size(file_stats.st_size)
        print(f"Saved interpolation coeffs in file CDF_coeffs.h5 with size {file_size}")
