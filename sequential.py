import numpy as np
import h5py
import scipy.interpolate as spip
from numbers import Real, Integral
import numba as nb

from time import process_time

from main import compute_BH_evolution, load_CDF_data
from predict_size import predict_size


if __name__ == "__main__":
    CDF_data = load_CDF_data("CDF_data.h5")
    CDF_vals = CDF_data[:, 0]
    x_vals = CDF_data[:, 1]

    inv_CDF_interp = spip.CubicSpline(CDF_vals, x_vals)

    init_J = 0.
    eps = 1e-4

    for init_M in range (10, 10001, 1):
        N = predict_size(init_M)
        print(init_M, N)
        for _ in range(100000):
            rands = np.random.uniform(
                size=N,
                low=CDF_vals[0],
                high=CDF_vals[-1],
            )
            changes_array = inv_CDF_interp(rands)
            rands_array = np.random.rand(N)

            t1 = process_time()
            M, J, a, n, extremal, path = compute_BH_evolution(init_M*1., init_J, eps, changes_array, rands_array)
            t2 = process_time()

            with open("./seq_result.csv", "a") as f:
                f.write(f"{init_M},{init_J},{M},{J},{a},{n},{t2-t1:.3e}\n")
