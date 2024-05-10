import numpy as np
from scipy.special import zeta

from numbers import Real

def P_c(M_ext: Real, A: Real = None) -> Real:
    """
    Calculate the probability distribution of the mass of an extremal 
    primordial black hole. Implements Eq. (13) of arXiv:2403.04054.

    Args:
    M_ext (Real): The external mass parameter.
    A (Real, optional): Normalization factor. Defaults to None.

    Returns:
    Real: The probability distribution of the mass.

    Notes:
    - Requires numpy to be imported.
    - If A is not provided, a default normalization factor is used.
    """
    γ = np.pi**4 / (30 * zeta(3))
    κ = 8 * np.sqrt(np.pi) / γ
    M = 0.5 * (M_ext + np.sqrt(M_ext**2 + γ / (2 * np.pi)))
    ν = np.log(1 + 2 / M**2 - 1 / M**4)
    expr = np.floor(M_ext**2)
    P = κ * M_ext * np.sqrt(ν) * (1 - expr / M**2)**2 * np.exp(-ν * expr**2)
    if A is None: A = 0.21129914651044093 # Normalization factor
    return P / A
