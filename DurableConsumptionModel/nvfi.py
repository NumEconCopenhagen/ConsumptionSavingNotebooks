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
def obj_keep(c,db,m,inv_w,par):
    """ evaluate bellman equation """

    penalty = 0
    c_min = np.fmin(m/2,1e-8)
    if c < c_min:
        penalty += 1000*(c-c_min)**2
        c = c_min
    if c > m:
        penalty += 1000*(c-m)**2
        c = m

    # a. end-of-period assets
    a = m-c
    
    # b. continuation value
    inv_w = linear_interp.interp_1d(par.grid_a,inv_w,a)
    w = -1/inv_w

    # c. total value
    value_of_choice = utility.func(c,db,par) + w

    return -value_of_choice + penalty # we are minimizing

# b. create optimizer
opt_keep = golden_section_search.create_optimizer(obj_keep)

@njit(parallel=True)
def solve_keep(t,sol,par):
    """solve bellman equation for keepers using nvfi"""

    # unpack
    inv_v = sol.inv_v_keep[t]
    c = sol.c_keep[t]

    # loop over outer states
    for ip in prange(par.Np):
        for idb in range(par.Ndb):
            
            # outer states
            db = par.grid_db[idb]

            # loop over m state
            for im in range(par.Nm):
                
                # a. cash-on-hand
                m = par.grid_m[im]
                
                # b. optimal choice
                c_low = np.fmin(m/2,1e-8)
                c_high = m
                c[ip,idb,im] = opt_keep(c_low,c_high,par.tol,db,m,sol.inv_w[t,ip,idb],par)

                # d. optimal value
                v = -obj_keep(c[ip,idb,im],db,m,sol.inv_w[t,ip,idb],par)
                inv_v[ip,idb,im] = -1/v

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
    db = d
    
    # b. value-of-choice
    inv_v_keep = linear_interp.interp_2d(par.grid_db,par.grid_m,inv_v_keep,db,m)
    
    return -inv_v_keep # we are minimizing

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
    for ip in prange(par.Np):
            
        # loop over m state
        for ix in range(par.Nx):
            
            # i. cash-on-hand
            x = par.grid_x[ix]
            
            # ii. optimal choice
            db_low = np.fmin(x/2,1e-8)
            db_high = np.fmin(x,par.db_max)
            d[ip,ix] = opt_adj(db_low,db_high,par.tol,x,sol.inv_v_keep[t,ip],par)

            # iii. optimal value
            m = x - d[ip,ix]
            c[ip,ix] = linear_interp.interp_2d(par.grid_db,par.grid_m,sol.c_keep[t,ip],d[ip,ix],m)
            inv_v[ip,ix] = -obj_adj(d[ip,ix],x,sol.inv_v_keep[t,ip],par)

