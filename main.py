import argparse
import numpy as np
import scipy.interpolate as spip
from numbers import Real, Integral
import numba as nb

import os
from time import process_time


def load_CDF_data(filename: str):
    CDF_data = np.loadtxt(filename, delimiter=",")
    return CDF_data

def predict_size(M: Real, safe: bool = True) -> Integral:
    log_log_intercept = 0.6643894764405784
    log_log_coef = 2.0030406770689377

    log_M = np.log10(M)
    log_N_predict = log_log_intercept + log_log_coef * log_M
    N_predict = 10**log_N_predict

    if safe: N_predict *= 2

    N_predict = int(np.ceil(N_predict))
    return N_predict

@nb.njit(fastmath = True)
def compute_BH_evolution(
    M_init: float,
    J_init: float,
    M_final: float,
    changes,
    rands,
    eps: float = 1,
):
    M = M_init
    J = J_init
    i = 0

    while M >= M_final:
        a_star = J / M**2.

        if np.abs(a_star) >= 1:
            a_star = np.sign(a_star) * 1.
            break

        rho_plus = 1 / 2 + eps * (- a_star + a_star * np.abs(a_star) / 2)

        T = np.sqrt(1 - a_star**2) / (4 * np.pi * M * (1 + np.sqrt(1 - a_star**2)))

        change = changes[i]

        M = M - change * T

        if rho_plus < rands[i]:
            J = J - 1
        else:
            J = J + 1

        i += 1

    return M, J, a_star, i


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ExtremalBlackHoles",
        description="Compute the evolution of PBHs",
    )
    parser.add_argument("-e", "--eps", type=float, default=1.)
    args = parser.parse_args()

    print(args.eps)

    CDF_data = load_CDF_data(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "CDF_data.csv",
    ))
    CDF_vals = CDF_data[:, 0]
    x_vals = CDF_data[:, 1]

    inv_CDF_interp = spip.CubicSpline(CDF_vals, x_vals)

    M_init = 100.
    J_init = 0.
    
    M_final = 1.

    eps = args.eps

    N = predict_size(init_M)

    for _ in range(10000):
        changes_array = inv_CDF_interp(np.random.rand(N))
        rands_array = np.random.rand(N)

        t1 = process_time()
        M, J, a_star, n = compute_BH_evolution(
            M_init, J_init,
            M_final,
            changes_array, rands_array,
            eps,
        )
        t2 = process_time()

        with open(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            f"results/test_M_{int(init_M)}.csv",
        ), "a") as f:
            f.write(f"{M_init},{J_init},{M},{J},{a_star},{n},{t2-t1}\n")

