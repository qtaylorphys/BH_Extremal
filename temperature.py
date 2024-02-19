import numpy as np
import numba as nb

from numbers import Real
from typing import Callable, Tuple, Any

@nb.njit(fastmath = True)
def Schwarzschild_T(M: Real, *_: Tuple[Any]) -> Real:
    """
    Computes the Hawking temperature for the Schwarzschild spacetime
    
    Parameters:
    M (Real): the mass of the black hole in Planck masses

    Returns:
    Real: the Hawking temperature
    """
    T = 1 / (8 * np.pi * M)
    return T

@nb.njit(fastmath = True)
def Kerr_T(M: Real, a_star: Real, *_: Tuple[Any]) -> Real:
    """
    Computes the Hawking temperature for the Kerr spacetime
    
    Parameters:
    M (Real): the mass of the black hole in Planck masses
    a_star (Real): the angular momentum of the black hole divided by the 
        mass squared (a_star = J / M**2)

    Returns:
    Real: the Hawking temperature
    """
    T = np.sqrt(1 - a_star**2) / (4 * np.pi * M * (1 + np.sqrt(1 - a_star**2)))
    return T

def Hawking_temperature(spacetime: str) -> Callable:
    """
    Returns a function for computing the Hawking temperature of a spacetime

    Parameters:
    spacetime (str): the spacetime of the black hole; should be one of 
        "Schwarzschild", "Kerr"

    Returns:
    Callable: a function to compute the Hawking temperature
    """
    if spacetime == "Schwarzschild":
        return Schwarzschild_T
    elif spacetime == "Kerr":
        return Kerr_T
    else:
        raise NotImplementedError("This spacetime is not supported yet")

if __name__ == "__main__":
    pass
