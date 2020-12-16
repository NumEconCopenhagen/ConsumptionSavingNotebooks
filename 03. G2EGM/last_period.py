import numpy as np
from numba import njit

import utility

# consav
from consav import linear_interp # for linear interpolation

@njit
def solve(sol,par,G2EGM=True):

    # unpack
    t = par.T-1
    c = sol.c[t]
    d = sol.d[t]
    inv_v = sol.inv_v[t]
    inv_vm = sol.inv_vm[t]
    if G2EGM:
        inv_vn = sol.inv_vn[t]

    for i_n in range(par.Nn):
    
        # i. states
        m  = par.grid_m
        n  = par.grid_n[i_n]
                
        # ii. consume everything
        d[i_n,:] = 0
        c[i_n,:] = m+n

        # iii. value function
        v = utility.func(c[i_n,:],par)
        inv_v[i_n,:] = -1.0/v
        
        # iv. value function derivatives
        vm = utility.marg_func(c[i_n,:],par)
        inv_vm[i_n,:] = 1.0/vm
        if G2EGM:
            inv_vn[i_n,:] = inv_vm[i_n,:]