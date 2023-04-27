import numpy as np
from mpmath import polylog, zeta
from scipy.optimize import root

from numbers import Real

import pandas as pd

def CDF(x: Real) -> Real:
    if isinstance(x, (list, np.ndarray)): x = x[0]
    
    cdf = (
        x**2 * np.log(1 - np.exp(-x))
        - 2 * (x * polylog(2, np.exp(-x)) + (polylog(3, np.exp(-x)) - zeta(3)))
    ) / (2 * zeta(3))
    return float(cdf)

def invert_CDF(x: Real, val: Real) -> Real:
    return CDF(x) - val

def find_x_from_CDF(val: Real) -> Real:
    x0 = 5e-1

    res = root(
        invert_CDF,
        x0,
        args=(val),
        method='lm',
        jac=None,
        tol=1e-9,
        callback=None,
        options=None,
    )

    return res.x[0]

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    cdffunction = pd.read_csv('cdffunction.csv', header = None)
    from scipy import interpolate
    y_points = np.linspace(0.000001,12,10001)#Number is 10001 instead of 10k due to indexing differences between mathematica and python

    def dist(x):
        invfunction = interpolate.splrep(cdffunction, y_points)
        return interpolate.splev(x, invfunction)
    
    N = 49000 #Number of time steps
    Nmc = 1000 #Number of Monte Carlo Sims (number of black holes)
    prob = rng.random(size = (Nmc,N))
    # change = dist(prob)

    mass_i = 100 #in Planck Masses
    Mass = np.zeros((Nmc+1,N+1))
    J = np.zeros((Nmc+1,N+1))
    astar = np.zeros((Nmc+1,N+1))
    Mass[:,0] = mass_i
    rand = rng.random(size = (Nmc,N))

    for n in range(Nmc):
        print("BH ", n)
        for i in range(N):
            # print(i)
            j = J[n,i]
            m = Mass[n,i]
            # print(j, m)
            if m < 1:
                Mass[n,i:] = 1
                break
            else:
                astar[n,i] = abs(j/m/m)
                if abs(j/m/m) > 1:
                    astar[n,i-1:] = 1
                    J[n,i-1:] = j
                    Mass[n,i-1:] = m
                    break
                rho = 1/2*(1-2*j/m**2)**2
                temp = 1/8/np.pi/m #schwarzchild temperature

                change_new = dist(prob[n, i])

                Mass[n,i+1] = m-change_new*temp 
                if rho > rand[n,i]:
                    J[n,i+1] = j-change_new*temp
                else:
                    J[n,i+1] = j+change_new*temp

                if i == 46424 or i == 46425 or i == 46426:
                    print(rho, temp, change_new, m, Mass[n,i], j, J[n,i], prob[n, i], rand[n,i])
        
        break

    J_old = J[0, :]
    M_old = Mass[0, :]
    
    init_M = 100
    M = init_M

    init_J = 0
    J = init_J

    n = 0
    i = 0
    print()
    while M >= 1:
        # print(i)
        # print(np.abs(J_old[i] - J))
        # print(np.abs(M_old[i] - M))

        a_star = J / M**2
        if a_star > 1:
            a_star = 1
            break

        rho = 1 / 2 * (1 - 2 * a_star)**2
        T = 1 / (8 * np.pi * M)

        probs = prob[n, i]
        rands = rand[n, i]
        change = dist(probs)

        if i == 46424 or i == 46425 or i == 46426:
            print(rho, T, change, M, M_old[i], J, J_old[i], prob[n, i], rand[n, i])

        M = M - change * T

        if rho > rands:
            J = J - change * T
        else:
            J = J + change * T

        i += 1


            
    #Mass[Mass < 1] = 1;
    #astar[astar > 1] = 1;
    # Mass = np.delete(Mass, -1, 0)
    # J = np.delete(J, -1, 0)
    # astar = np.delete(astar,-1,0)