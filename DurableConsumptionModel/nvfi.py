import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp # for linear interpolation
from consav import golden_section_search

# local modules
import utility

########
# keep #
########

# a. define objective function
@njit
def obj_keep(c,n,m,inv_w,par):
    """ evaluate bellman equation """

    # a. end-of-period assets
    a = m-c
    
    # b. continuation value
    w = -1.0/linear_interp.interp_1d(par.grid_a,inv_w,a)

    # c. total value
    value_of_choice = utility.func(c,n,par) + w

    return -value_of_choice # we are minimizing

# b. create optimizer
opt_keep = golden_section_search.create_optimizer(obj_keep)

@njit(parallel=True)
def solve_keep(t,sol,par):
    """solve bellman equation for keepers using nvfi"""

    # unpack
    inv_v = sol.inv_v_keep[t]
    c = sol.c_keep[t]

    # loop over outer states
    for i_p in prange(par.Np):
        for i_n in range(par.Nn):
            
            # outer states
            n = par.grid_n[i_n]

            # loop over m state
            for i_m in range(par.Nm):
                
                # a. cash-on-hand
                m = par.grid_m[i_m]
                
                # b. optimal choice
                c_low = np.fmin(m/2,1e-8)
                c_high = m
                c[i_p,i_n,i_m] = opt_keep(c_low,c_high,par.tol,n,m,sol.inv_w[t,i_p,i_n],par)

                # c. optimal value
                v = -obj_keep(c[i_p,i_n,i_m],n,m,sol.inv_w[t,i_p,i_n],par)
                inv_v[i_p,i_n,i_m] = -1/v

#######
# adj #
#######

# a. define objective function
@njit
def obj_adj(d,x,inv_v_keep,par):
    """ evaluate bellman equation """

    # a. cash-on-hand
    m = x-d

    # b. durables
    n = d
    
    # c. value-of-choice
    return -linear_interp.interp_2d(par.grid_n,par.grid_m,inv_v_keep,n,m)  # we are minimizing

# b. create optimizer
opt_adj = golden_section_search.create_optimizer(obj_adj)

@njit(parallel=True)
def solve_adj(t,sol,par):
    """solve bellman equation for adjusters using nvfi"""

    # unpack
    inv_v = sol.inv_v_adj[t]
    d = sol.d_adj[t]
    c = sol.c_adj[t]

    # loop over outer states
    for i_p in prange(par.Np):
            
        # loop over x state
        for i_x in range(par.Nx):
            
            # a. cash-on-hand
            x = par.grid_x[i_x]
            
            # b. optimal choice
            d_low = np.fmin(x/2,1e-8)
            d_high = np.fmin(x,par.n_max)
            d[i_p,i_x] = opt_adj(d_low,d_high,par.tol,x,sol.inv_v_keep[t,i_p],par)

            # c. optimal value
            m = x - d[i_p,i_x]
            c[i_p,i_x] = linear_interp.interp_2d(par.grid_n,par.grid_m,sol.c_keep[t,i_p],d[i_p,i_x],m)
            inv_v[i_p,i_x] = -obj_adj(d[i_p,i_x],x,sol.inv_v_keep[t,i_p],par)

