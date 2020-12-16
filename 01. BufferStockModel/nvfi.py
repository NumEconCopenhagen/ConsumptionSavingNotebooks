import numpy as np
from numba import njit, prange

 # consav
from consav import linear_interp # for linear interpolation
from consav import golden_section_search # for optimization in 1D

# local modules
import utility

# a. define objective function
@njit
def obj_bellman(c,m,interp_w,par):
    """ evaluate bellman equation """

    # a. end-of-period assets
    a = m-c
    
    # b. continuation value
    w = linear_interp.interp_1d(par.grid_a,interp_w,a)

    # c. total value
    value_of_choice = utility.func(c,par) + w

    return -value_of_choice # we are minimizing

# b. solve bellman equation        
@njit(parallel=True)
def solve_bellman(t,sol,par):
    """solve bellman equation using nvfi"""

    # unpack (this helps numba optimize)
    v = sol.v[t]
    c = sol.c[t]

    # loop over outer states
    for ip in prange(par.Np): # in parallel

        # loop over cash-on-hand
        for im in range(par.Nm):
            
            # a. cash-on-hand
            m = par.grid_m[im]

            # b. optimal choice
            c_low = np.fmin(m/2,1e-8)
            c_high = m
            c[ip,im] = golden_section_search.optimizer(obj_bellman,c_low,c_high,args=(m,sol.w[ip],par),tol=par.tol)

            # note: the above finds the minimum of obj_bellman in range [c_low,c_high] with a tolerance of par.tol
            # and arguments (except for c) as specified 

            # c. optimal value
            v[ip,im] = -obj_bellman(c[ip,im],m,sol.w[ip],par)

