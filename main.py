import argparse
import numpy as np
import os
import h5py

from pbh import PrimordialBlackHole


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

    pbh = PrimordialBlackHole(
        args.spacetime,
        args.initial_mass, args.final_mass,
        args.initial_mom,
        args.eps,
    )

    N_PBH = 50000000
    zfill_len = int(np.log10(N_PBH)) + 1

    for i in range(N_PBH):
        if i % 1000000 == 0: print(i)
        
        pbh.evolve()

        if np.isnan(pbh.M_end) or np.isnan(pbh.J_end):
            print(i)
            print(pbh.N)
            print(pbh.M_init)
            print(pbh.J_init)
            print(pbh.M_end)
            print(pbh.J_end)
            print(pbh.a_star_end)
            print(pbh.n_steps)
            print(pbh.extremal)
            print(pbh.path)
            break

        with open(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            f"results/test_M_{int(pbh.M_init)}.csv",
        ), "a") as f:
            f.write(f"{pbh.M_init},{pbh.J_init},{pbh.M_end},{pbh.J_end},{pbh.a_star_end},{pbh.n_steps},{int(pbh.extremal)},{pbh.computation_time:.3e}\n")

            if pbh.path is not None:
                with h5py.File(f"./path{str(i).zfill(zfill_len)}.h5", 'w') as f:
                    f.create_dataset(
                        "path",
                        data=pbh.path,
                        compression="gzip",
                        compression_opts=9,
                    )
