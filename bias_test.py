import numpy as np
import scipy.interpolate as spip
from numbers import Real, Integral
import numba as nb

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

    if safe: N_predict *= 5

    N_predict = int(np.ceil(N_predict))
    return N_predict

@nb.njit(fastmath = True)
def compute_BH_evolution(
    M_init,
    J_init,
    M_final,
    changes,
    rands,
    eps,
):
    M = M_init
    J = J_init
    i = 0
    eps = eps #Factor that will turn on the bias
    
    while M >= M_final:
        a_star = J / M**2.

        if a_star >= 1:
            a_star = 1.
            break
        if a_star <= -1:
            a_star = -1.
            break

        
        #rho_plus = 1/2 - a_star + a_star * np.abs(a_star) / 2
        rho_plus = 1 / 2  + eps * (-a_star + a_star * np.abs(a_star) / 2)

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
    CDF_data = load_CDF_data("/home/qxt42/Documents/PBH_Extremal/Data_Files/CDF_data.csv")
    CDF_vals = CDF_data[:, 1]
    x_vals = CDF_data[:, 0]

    inv_CDF_interp = spip.CubicSpline(CDF_vals, x_vals)

    init_M = 100.
    init_J = 0.

    eps=0.5 
    
    M_final = 1.

    N = predict_size(100)
    changes_array = inv_CDF_interp(np.random.rand(N))
    rands_array = np.random.rand(N)
    #print(N)
    # compile the function
    # _, _, _, _ = compute_BH_evolution(100., 0., M_final, changes_array, rands_array)

    for _ in range(2000):
        N = predict_size(init_M)

        changes_array = inv_CDF_interp(np.random.rand(N))
        rands_array = np.random.rand(N)

        
        
        t1 = process_time()
        M, J, a, n = compute_BH_evolution(init_M, init_J, M_final, changes_array, rands_array, eps)
        t2 = process_time()

        b=0.5*100 
        with open(f"/home/qxt42/Documents/PBH_Extremal/bias{int(b)}_M_{int(init_M)}.txt", "a") as f:
           f.write(f"{init_M},{init_J},{M},{J},{a},{n},{t2-t1}\n")

        #with open(f"/home/qxt42/Documents/PBH_Extremal/test_ang_{int(init_M)}.txt", "a") as f:
        #    f.write(ang)

