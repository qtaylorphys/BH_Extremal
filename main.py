import argparse
import numpy as np
import h5py
import scipy.interpolate as spip
from numbers import Real, Integral
from nptyping import NDArray
import numba as nb

import os
from time import process_time

from interpolate import cubic_spline


def load_CDF_data(filename: str):
    with h5py.File(filename,'r') as f:
        CDF_data = f["CDF_data"][:]
    return CDF_data

@nb.njit(fastmath = True)
def predict_size(M: Real, safe: bool = True) -> Integral:
    log_log_intercept = 0.6643894764405784
    log_log_coef = 2.0030406770689377

    log_M = np.log10(M)
    log_N_predict = log_log_intercept + log_log_coef * log_M
    N_predict = 10**log_N_predict

    if safe: N_predict *= 1.2

    N_predict = int(np.ceil(N_predict))
    return N_predict

@nb.njit(fastmath = True)
def compute_interp(f, x):
    y = x.argsort()
    z = f(x[y])
    i = np.empty_like(y)
    i[y] = np.arange(y.size)
    return z[i]

@nb.njit(fastmath = True)
def compute_BH_evolution(
    M_init: float,
    J_init: float,
    M_final: float,
    changes: NDArray,
    rands: NDArray,
    eps: float = 1,
    return_path: bool = False,
):
    M = M_init
    J = J_init
    i = 0
    extremal = False

    if return_path:
        M_path = []
        J_path = []
        a_star_path = []
        rho_plus_path = []
        T_path = []

    while M >= M_final:
        a_star = J / M**2.

        if np.abs(a_star) >= 1:
            a_star = np.sign(a_star) * 1.
            extremal = True
            
        if return_path:
            M_path.append(M)
            J_path.append(J)
            a_star_path.append(a_star)

        rho_plus = 1 / 2 + eps * (- a_star + a_star * np.abs(a_star) / 2)
        T = np.sqrt(1 - a_star**2) / (4 * np.pi * M * (1 + np.sqrt(1 - a_star**2)))

        if return_path:
            rho_plus_path.append(rho_plus)
            T_path.append(T)

        if extremal: break

        change = changes[i]

        M = M - change * T

        if rho_plus < rands[i]:
            J = J - 1
        else:
            J = J + 1

        i += 1

    extremal *= 1
    n_steps = i + extremal

    if return_path:
        path = np.stack((
            np.asarray(M_path),
            np.asarray(J_path),
            np.asarray(a_star_path),
            np.asarray(rho_plus_path),
            np.asarray(T_path),
            rands[:n_steps],
            changes[:n_steps]),
            axis=-1,
        )
    else:
        path = None

    return M, J, a_star, n_steps, extremal, path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the evolution of PBHs",
    )
    parser.add_argument("-e", "--eps", type=float, default=1.)
    args = parser.parse_args()

    CDF_data = load_CDF_data(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "CDF_data.h5",
    ))
    CDF_vals = CDF_data[:, 0]
    x_vals = CDF_data[:, 1]

    inv_CDF_interp = spip.CubicSpline(CDF_vals, x_vals)

    new_cs = cubic_spline(CDF_vals, x_vals)

    M_init = 1000.
    J_init = 0.
    
    M_final = 1.

    eps = args.eps

    N = predict_size(M_init)

    N_PBH = 100
    zfill_len = int(np.log10(N_PBH)) + 1

    for i in range(N_PBH):
        print(i)
        rands = np.random.uniform(
            size=N,
            low=CDF_vals[0],
            high=CDF_vals[-1],
        )

        t1 = process_time()
        changes_array = inv_CDF_interp(rands)
        t2 = process_time()
        # changes_array2 = compute_interp(new_cs, rands)

        # print(np.max(np.abs(changes_array - changes_array2)))
        # j = np.argmax(np.abs(changes_array - changes_array2))
        # print(rands_array0[j], changes_array[j], changes_array2[j])
        # print(t3 - t2, t2 - t1)
        # print()
        print(t2 - t1)

        rands_array = np.random.rand(N)

        t1 = process_time()
        M, J, a_star, n, extremal, path = compute_BH_evolution(
            M_init, J_init,
            M_final,
            changes_array, rands_array,
            eps,
            False,
        )
        t2 = process_time()

        with open(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            f"results/test_M_{int(M_init)}.csv",
        ), "a") as f:
            f.write(f"{M_init},{J_init},{M},{J},{a_star},{n},{t2-t1:.3e}\n")

            if path is not None:
                with h5py.File(f"./path{str(i).zfill(zfill_len)}.h5", 'w') as f:
                    f.create_dataset(
                        "path",
                        data=path,
                        compression="gzip",
                        compression_opts=9,
                    )
