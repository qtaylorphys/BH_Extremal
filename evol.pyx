import cython

cimport numpy as np
import numpy as np
from libc.math cimport cos, sqrt, ceil, M_PI, fabs
from libc.stdio cimport printf
from libc.stdlib cimport rand, RAND_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(True)
cpdef tuple compute_BH_evolution(
    double M_init,
    double J_init,
    double eps,
    double[:] changes,
    double[:] rands,
):
    cdef double M = M_init
    cdef double J = J_init
    cdef int i = 0

    cdef double a_star
    cdef double rho
    cdef double T
    cdef double change

    while M >= 1.:
        a_star = J / M**2.

        if fabs(a_star) > 1 - eps:
            a_star = 1.
            break

        rho = 1 / 2 * (1 - 2 * a_star)**2

        T = sqrt(1 - a_star**2) / (4 * M_PI * M * (1 + sqrt(1 - a_star**2)))

        change = changes[i]

        M = M - change * T

        if rho > rands[i]:
            J = J - change * T
        else:
            J = J + change * T

        i += 1

    return M, J, a_star, i
