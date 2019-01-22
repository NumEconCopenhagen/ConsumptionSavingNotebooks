import numpy as np
from numba import njit, prange

 # consav
from consav import linear_interp # for linear interpolation

# local modules
import trans

@njit(parallel=True)
def lifecycle(sim,sol,par):
    """ simulate full life-cycle """

    # unpack
    p = sim.p
    db = sim.db
    m = sim.m
    x = sim.x
    c = sim.c
    d = sim.d
    a = sim.a
    discrete = sim.discrete
    
    for t in range(par.simT):

        for i in prange(par.simN):
            
            # a. beginning of period states
            if t == 0:
                p[t,i] = 1.0
                db[t,i] = 0.0
                m[t,i] = 1.0
                x[t,i] = trans.x_plus_func(m[t,i],db[t,i],par)
            else:
                p[t,i] = trans.p_plus_func(p[t-1,i],sim.psi[t,i],par)
                db[t,i] = trans.db_plus_func(d[t-1,i],par)
                m[t,i] = trans.m_plus_func(a[t-1,i],p[t,i],sim.xi[t,i],par)
                x[t,i] = trans.x_plus_func(m[t,i],db[t,i],par)
                                
            # b. discrete choice
            inv_v_keep = linear_interp.interp_3d(par.grid_p,par.grid_db,par.grid_m,sol.inv_v_keep[t],p[t,i],db[t,i],m[t,i])
            inv_v_adj = linear_interp.interp_2d(par.grid_p,par.grid_x,sol.inv_v_adj[t],p[t,i],x[t,i])    

            # c. continuous choices
            if inv_v_adj > inv_v_keep:

                discrete[t,i] = 1
                
                d[t,i] = linear_interp.interp_2d(
                    par.grid_p,par.grid_x,sol.d_adj[t],
                    p[t,i],x[t,i])

                c[t,i] = linear_interp.interp_2d(
                    par.grid_p,par.grid_x,sol.c_adj[t],
                    p[t,i],x[t,i])

                tot = d[t,i]+c[t,i]
                if tot > x[t,i]: 
                    d[t,i] *= x[t,i]/tot
                    c[t,i] *= x[t,i]/tot
                    a[t,i] = 0.0
                else:
                    a[t,i] = x[t,i] - tot
            
            else: 
                
                discrete[t,i] = 0

                d[t,i] = db[t,i]

                c[t,i] = linear_interp.interp_3d(
                    par.grid_p,par.grid_db,par.grid_m,sol.c_keep[t],
                    p[t,i],db[t,i],m[t,i])

                if c[t,i] > m[t,i]: 
                    c[t,i] = m[t,i]
                    a[t,i] = 0.0
                else:
                    a[t,i] = m[t,i] - c[t,i]