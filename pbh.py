from abc import ABC
import numpy as np
import numba as nb
import os
import scipy.interpolate as spip

from numbers import Real

from utils import timing, load_CDF_data
from predict_size import predict_size
from temperature import Hawking_temperature

from numbers import Real, Integral
from typing import Tuple, Callable
from nptyping import NDArray


@nb.njit(fastmath = True)
def compute_interp(f, x):
    y = x.argsort()
    z = f(x[y])
    i = np.empty_like(y)
    i[y] = np.arange(y.size)
    return z[i]

@nb.njit(fastmath = True)
def compute_rho(a_star: Real, eps: Real = 1.) -> Real:
    """
    Compute the probability that the black hole spin will increase

    Parameters:
    a_star (Real): the angular momentum of the black hole divided by the 
        mass squared (a_star = J / M**2)

    eps (Real): tuning parameter used to control the deviation from even
        probability (1/2); use eps = 1. for production results

    Returns:
    Real: probability
    """
    rho = 1 / 2 + eps * (- a_star + a_star * np.abs(a_star) / 2)
    return rho

@timing
@nb.njit(fastmath = True)
def compute_evolution(
    M_init: float,
    M_final: float,
    J_init: Integral,
    compute_T: Callable,
    changes: NDArray,
    rands: NDArray,
    eps: Real = 1,
    return_path: bool = False,
) -> Tuple:
    """
    Compute the evolution of a (primordial) black hole as it evaporates due 
    to Hawking radiation

    Parameters:
    M_init (Real): the initial mass of the black hole (in Planck masses)
    J_init (Integral): the initial angular momentum of the black hole (in 
        units of h-bar)
    M_final (Real): the final mass of the black hole (in units of Planck 
        masses)
    compute_T (Callable): function which computes the Hawking temperature
    changes (NDArray):
    rands (NDArray):
    eps (Real): tuning parameter used to control the deviation from even
        probability (1/2); use eps = 1. for production results
    return path (bool): if True, will return the evolution of M, J, a_star, 
        rho_plus, T, as well as the rands and changes arrays

    Returns:
    Real: the final mass of the black hole
    Real: the final angular momentum of the black hole
    Real: the final value of a_star (equal to +/- 1 if extremal)
    Integral: the number of random steps taken
    bool: whether the black hole became extremal
    path: if return_path is True, contains the values of M, J, a_star, 
        rho_plus, T, rands, changes as an array of shape (n_steps, 7); 
        else None
    """
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

        rho_plus = compute_rho(a_star, eps)
        T = compute_T(M, a_star)

        if return_path:
            M_path.append(M)
            J_path.append(J)
            a_star_path.append(a_star)
            rho_plus_path.append(rho_plus)
            T_path.append(T)

        if extremal: break

        M = M - changes[i] * T

        if rho_plus < rands[i]:
            J = J - 1
        else:
            J = J + 1
        
        i += 1
        
        if i >= rands.size:
            raise IndexError

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


class PrimordialBlackHole(ABC):
    def __init__(
        self,
        spacetime: str,
        M_init: Real,
        M_final: Real = 1,
        J_init: Real = 0,
        eps: Real = 1,
        save_path: bool = False,
    ) -> None:
        self.spacetime = spacetime
        self.M_init = M_init
        self.M_final = M_final
        self.J_init = J_init
        self.eps = eps
        self.save_path = save_path

        self.rands_array = np.array([])
        self.changes_array = np.array([])

        self.M_end = None
        self.J_end = None
        self.a_star_end = None
        self.n_steps = None
        self.extremal = None
        self.path = None

        self.computation_time = None
        
        self.N = predict_size(self.M_init)
        self.temperature_f = Hawking_temperature(self.spacetime)

        CDF_data = load_CDF_data(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "CDF_data.h5",
        ))
        self.CDF_values = CDF_data[:, 0]
        self.x_values = CDF_data[:, 1]
        self.inv_CDF_interp = spip.CubicSpline(self.CDF_values, self.x_values)

    def _construct_rands_array(self) -> None:
        current_size = self.rands_array.size
        delta_size = self.N - current_size
        self.rands_array = np.append(
            self.rands_array,
            np.random.default_rng().uniform(size=delta_size),
        )
        
    def _construct_changes_array(self) -> None:
        current_size = self.changes_array.size
        delta_size = self.N - current_size
        self.CDF_rands_array = np.random.default_rng().uniform(
            size=delta_size,
            low=self.CDF_values[0] + 1e-18,
            high=self.CDF_values[-1],
        )
        self.changes_array = np.append(
            self.changes_array,
            self.inv_CDF_interp(self.CDF_rands_array)
        )

    def evolve(self) -> None:
        self.attempts = 1
        while True:
            self._construct_rands_array()
            self._construct_changes_array()

            try:
                (M, J, a_star, n, extremal, path), time = compute_evolution(
                    self.M_init, self.M_final,
                    self.J_init,
                    self.temperature_f,
                    self.changes_array, self.rands_array,
                    self.eps,
                    self.save_path,
                )
                break
            except IndexError:
                extra_factor = 1.5
                self.N = int(np.ceil(extra_factor * self.N))
                self.attempts += 1

        self.M_end = M
        self.J_end = J
        self.a_star_end = a_star
        self.n_steps = n
        self.extremal = extremal
        self.path = path
        self.computation_time = time


if __name__ == "__main__":
    pass
