from numba import njit, prange
from consav import golden_section_search

# local modules
import utility

# a. objective
@njit
def obj_last_period(d,x,par):
    """ objective function in last period """
    
    # implied consumption (rest)
    c = x-d

    return -utility.func(c,d,par)

# b. create optimizer
opt_last_period = golden_section_search.create_optimizer(obj_last_period)

@njit(parallel=True)
def solve(t,sol,par):
    """ solve the problem in the last period """

    # unpack
    inv_v_keep = sol.inv_v_keep[t]
    c_keep = sol.c_keep[t]
    inv_v_adj = sol.inv_v_adj[t]
    d_adj = sol.d_adj[t]
    c_adj = sol.c_adj[t]

    # a. keep
    for ip in prange(par.Np):
        for idb in range(par.Ndb):
            for im in range(par.Nm):
                            
                # a. states
                _p = par.grid_p[ip]
                db = par.grid_db[idb]
                m = par.grid_m[im]

                if m == 0: # forced c = 0 
                    c_keep[ip,idb,im] = 0
                    inv_v_keep[ip,idb,im] = 0
                    continue
                
                # b. optimal choice
                c_keep[ip,idb,im] = m

                # c. optimal value
                v_keep = utility.func(c_keep[ip,idb,im],db,par)
                inv_v_keep[ip,idb,im] = -1.0/v_keep

    # b. adj
    for ip in prange(par.Np):
        for ix in range(par.Nx):
            
            # i. states
            _p = par.grid_p[ip]
            x = par.grid_x[ix]

            if x == 0: # forced c = d = 0
                d_adj[ip,ix] = 0
                c_adj[ip,ix] = 0
                inv_v_adj[ip,ix] = 0
                continue

            # ii. optimal choices
            d_adj[ip,ix] = opt_last_period(0,x,par.tol,x,par)
            c_adj[ip,ix] = x-d_adj[ip,ix]

            # iii. optimal value
            v_adj = -obj_last_period(d_adj[ip,ix],x,par)
            inv_v_adj[ip,ix] = -1.0/v_adj