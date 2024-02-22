import numpy as np
import numba as nb

from numbers import Real, Integral

@nb.njit(fastmath = True)
def predict_size(M: Real, extra_factor: Real = 1.) -> Integral:
    """
    Predict the size of the arrays changes and rands needed for the 
    compute_BH_evolution function. The parameters within are derived in 
    the predict_size.ipynb notebook.

    Parameters:
    M (Real): the initial mass of the black hole (in Planck masses)
    extra_factor (Real): extra factor for multiplying the size that can 
        be provided by the user

    Returns:
    Integral: the predicted size for the arrays
    """
    log_log_intercept = 0.8156006413788646
    log_log_coef = 1.9156352509417214

    log_M = np.log10(M)
    log_N_predict = log_log_intercept + log_log_coef * log_M
    N_predict = 10**log_N_predict

    N_predict *= 1.2
    N_predict *= extra_factor

    N_predict = int(np.ceil(N_predict))
    return N_predict

