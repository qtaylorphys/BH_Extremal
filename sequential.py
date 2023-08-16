import numpy as np
import h5py
import scipy.interpolate as spip
from numbers import Real, Integral
import numba as nb

from time import process_time


def load_CDF_data(filename: str):
    with h5py.File(filename,'r') as f:
        CDF_data = f["CDF_data"][:]
    return CDF_data

def predict_size(M: Real, safe: bool = True) -> Integral:
    log_log_intercept = 0.6643894764405784
    log_log_coef = 2.0030406770689377

    log_M = np.log10(M)
    log_N_predict = log_log_intercept + log_log_coef * log_M
    N_predict = 10**log_N_predict

    if safe: N_predict *= 40

    N_predict = int(np.ceil(N_predict))
    return N_predict

@nb.njit(fastmath = True)
def compute_BH_evolution(
    M_init,
    J_init,
    eps,
    changes,
    rands,
):
    M = M_init
    J = J_init
    i = 0

    while M >= 1.:
        a_star = J / M**2.

        if np.abs(a_star) > 1 - eps:
            a_star = 1.
            break

        rho = 1 / 2 * (1 - 2 * a_star)**2

        T = np.sqrt(1 - a_star**2) / (4 * np.pi * M * (1 + np.sqrt(1 - a_star**2)))

        change = changes[i]

        M = M - change * T

        if rho > rands[i]:
            J = J - change * T
        else:
            J = J + change * T

        i += 1

    return M, J, a_star, i


if __name__ == "__main__":
    CDF_data = load_CDF_data("CDF_data.h5")
    CDF_vals = CDF_data[:, 0]
    x_vals = CDF_data[:, 1]

    inv_CDF_interp = spip.CubicSpline(CDF_vals, x_vals)

    init_J = 0.
    eps = 1e-4

    N = predict_size(100)
    changes_array = inv_CDF_interp(np.random.rand(N))
    rands_array = np.random.rand(N)

    # compile the function
    _, _, _, _ = compute_BH_evolution(100., 0., eps, changes_array, rands_array)

    for init_M in range (10, 10001, 10):
        N = predict_size(init_M)
        print(init_M, N)
        for _ in range(20):
            changes_array = inv_CDF_interp(np.random.rand(N))
            rands_array = np.random.rand(N)

            t1 = process_time()
            M, J, a, n = compute_BH_evolution(init_M, init_J, eps, changes_array, rands_array)
            t2 = process_time()

            with open("/home/dmihaylov/Extremal_BH/seq_result4.csv", "a") as f:
                f.write(f"{init_M},{init_J},{M},{J},{a},{n},{t2-t1}\n")

