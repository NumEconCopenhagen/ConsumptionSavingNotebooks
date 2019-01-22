import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp # for linear interpolation

# local modules
import utility
import trans

@njit
def obj(c,p,db,m,inv_v_plus_keep,inv_v_plus_adj,par):
    """ evaluate bellman equation """

    # a. end-of-period assets
    a = m-c
    
    # b. continuation value
    w = 0
    for ishock in range(par.Nshocks):
            
        # i. shocks
        psi = par.psi[ishock]
        psi_w = par.psi_w[ishock]
        xi = par.xi[ishock]
        xi_w = par.xi_w[ishock]

        # ii. next-period states
        p_plus = trans.p_plus_func(p,psi,par)
        db_plus = trans.db_plus_func(db,par)
        m_plus = trans.m_plus_func(a,p_plus,xi,par)
        x_plus = trans.x_plus_func(m_plus,db_plus,par)
                
        # iii. weight
        weight = psi_w*xi_w
        
        # iv. update
        inv_v_plus_keep_now = linear_interp.interp_3d(par.grid_p,par.grid_db,par.grid_m,inv_v_plus_keep,p_plus,db_plus,m_plus)    
        inv_v_plus_adj_now = linear_interp.interp_2d(par.grid_p,par.grid_x,inv_v_plus_adj,p_plus,x_plus)    
        
        v_plus_now = -np.inf
        if inv_v_plus_keep_now > inv_v_plus_adj_now and inv_v_plus_keep_now > 0:
            v_plus_now = -1.0/inv_v_plus_keep_now
        elif inv_v_plus_adj_now > 0:
            v_plus_now = -1.0/inv_v_plus_adj_now
        w += weight*par.beta*v_plus_now

    # c. total value
    value_of_choice = utility.func(c,db,par) + w

    return value_of_choice

@njit(parallel=True)
def solve_keep(t,sol,par):
    """solve bellman equation for keepers using vfi"""
    
    # unpack
    inv_v = sol.inv_v_keep[t]
    c = sol.c_keep[t]

    # keep: loop over outer states
    for ip in prange(par.Np):
        for idb in range(par.Ndb):
            
            # outer states
            p = par.grid_p[ip]
            db = par.grid_db[idb]

            # loop over m state
            for im in range(par.Nm):
                
                if im == 0:
                    c[ip,idb,im] = 0
                    inv_v[ip,idb,im] = 0
                    continue

                # a. cash-on-hand
                m = par.grid_m[im]
                
                # b. optimal choice
                c_low = np.fmin(m/2,1e-8)
                c_high = m
                
                v_max = -np.inf
                c_max = 0

                for ic in range(par.Nc_keep):
                    
                    c_now = c_low + par.grid_c_keep[ic]*(c_high-c_low)
                    v_now = obj(c_now,p,db,m,sol.inv_v_keep[t+1],sol.inv_v_adj[t+1],par)
                    if v_now > v_max:
                       v_max = v_now
                       c_max = c_now

                # c. optimal value
                c[ip,idb,im] = c_max
                inv_v[ip,idb,im] = -1.0/v_max

@njit(parallel=True)
def solve_adj(t,sol,par):
    """solve bellman equation for keepers using vfi"""

    # unpack
    inv_v = sol.inv_v_adj[t]
    d = sol.d_adj[t]
    c = sol.c_adj[t]

    # keep: loop over outer states
    for ip in prange(par.Np):
            
            # outer states
            p = par.grid_p[ip]

            # loop over x state
            for ix in range(par.Nx):
                
                if ix == 0:
                    c[ip,ix] = 0
                    d[ip,ix] = 0
                    inv_v[ip,ix] = 0
                    continue

                # a. cash-on-hand
                x = par.grid_x[ix]

                d_low = 0
                d_high = x

                v_max = -np.inf
                d_max = 0                
                c_max = 0                
                
                # loop over d choice
                for id in range(par.Nd_adj):
                    
                    d_now = d_low + par.grid_d_adj[id]*(d_high-d_low)
                    m = x-d_now
                    c_low = np.fmin(x/2,1e-8)
                    c_high = x

                    for ic in range(par.Nc_adj):

                        c_now = c_low + par.grid_c_adj[ic]*(c_high-c_low)
                        v_now = obj(c_now,p,d_now,m,sol.inv_v_keep[t+1],sol.inv_v_adj[t+1],par)
                        if v_now > v_max:
                            v_max = v_now
                            d_max = d_now
                            c_max = c_now

                # d. optimal value
                d[ip,ix] = d_max
                c[ip,ix] = c_max
                inv_v[ip,ix] = -1/v_max