import argparse
import numpy as np
import h5py
import scipy.interpolate as spip

import os
from time import process_time

from pbh import PrimordialBlackHole
from utils import load_CDF_data, timing
from interpolate import cubic_spline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the evolution of PBHs",
    )
    parser.add_argument("-s", "--spacetime", type=str, default="Kerr")
    parser.add_argument("-Minit", "--initial_mass", type=float, default=100.)
    parser.add_argument("-Mfinal", "--final_mass", type=float, default=1.)
    parser.add_argument("-Jinit", "--initial_mom", type=int, default=0.)
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

    pbh = PrimordialBlackHole(
        args.spacetime,
        args.initial_mass, args.final_mass,
        args.initial_mom,
        args.eps,
    )

    # N = predict_size(M_init)
    # temperature_f = Hawking_temperature(spacetime)

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

        # t1 = process_time()

        with open(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            f"results/test_M_{int(M_init)}.csv",
        ), "a") as f:
            f.write(f"{M_init},{J_init},{M},{J},{a_star},{n},{time:.3e}\n")

            if path is not None:
                with h5py.File(f"./path{str(i).zfill(zfill_len)}.h5", 'w') as f:
                    f.create_dataset(
                        "path",
                        data=path,
                        compression="gzip",
                        compression_opts=9,
                    )
