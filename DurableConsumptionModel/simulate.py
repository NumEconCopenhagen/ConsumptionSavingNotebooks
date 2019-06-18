import numpy as np
from numba import njit, prange

 # consav
from consav import linear_interp # for linear interpolation

# local modules
import trans
import utility

@njit(parallel=True)
def lifecycle(sim,sol,par):
    """ simulate full life-cycle """

    # unpack
    p = sim.p
    n = sim.n
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
                p[t,i] = trans.p_plus_func(sim.p0[i],sim.psi[t,i],par)
                n[t,i] = trans.n_plus_func(sim.d0[i],par)
                m[t,i] = trans.m_plus_func(sim.a0[i],p[t,i],sim.xi[t,i],par)
            else:
                p[t,i] = trans.p_plus_func(p[t-1,i],sim.psi[t,i],par)
                n[t,i] = trans.n_plus_func(d[t-1,i],par)
                m[t,i] = trans.m_plus_func(a[t-1,i],p[t,i],sim.xi[t,i],par)
            x[t,i] = trans.x_plus_func(m[t,i],n[t,i],par)

            optimal_choice(t,p[t,i],n[t,i],m[t,i],x[t,i],discrete[t,i:],d[t,i:],c[t,i:],a[t,i:],sol,par)
            
@njit            
def optimal_choice(t,p,n,m,x,discrete,d,c,a,sol,par):

    # a. discrete choice
    inv_v_keep = linear_interp.interp_3d(par.grid_p,par.grid_n,par.grid_m,sol.inv_v_keep[t],p,n,m)
    inv_v_adj = linear_interp.interp_2d(par.grid_p,par.grid_x,sol.inv_v_adj[t],p,x)    
    adjust = inv_v_adj > inv_v_keep
    
    # b. continuous choices
    if adjust:

        discrete[0] = 1
        
        d[0] = linear_interp.interp_2d(
            par.grid_p,par.grid_x,sol.d_adj[t],
            p,x)

        c[0] = linear_interp.interp_2d(
            par.grid_p,par.grid_x,sol.c_adj[t],
            p,x)

        tot = d[0]+c[0]
        if tot > x: 
            d[0] *= x/tot
            c[0] *= x/tot
            a[0] = 0.0
        else:
            a[0] = x - tot
            
    else: 
            
        discrete[0] = 0

        d[0] = n

        c[0] = linear_interp.interp_3d(
            par.grid_p,par.grid_n,par.grid_m,sol.c_keep[t],
            p,n,m)

        if c[0] > m: 
            c[0] = m
            a[0] = 0.0
        else:
            a[0] = m - c[0]

@njit            
def euler_errors(sim,sol,par):

    # unpack
    euler_error = sim.euler_error
    euler_error_c = sim.euler_error_c
    
    for i in prange(par.simN):
        
        discrete_plus = np.zeros(1)
        d_plus = np.zeros(1)
        c_plus = np.zeros(1)
        a_plus = np.zeros(1)

        for t in range(par.simT-1):

            constrained = sim.a[t,i] < par.euler_cutoff
            
            if constrained:

                euler_error[t,i] = np.nan
                euler_error_c[t,i] = np.nan
                continue

            else:

                RHS = 0.0
                for ishock in range(par.Nshocks):
                        
                    # i. shocks
                    psi = par.psi[ishock]
                    psi_w = par.psi_w[ishock]
                    xi = par.xi[ishock]
                    xi_w = par.xi_w[ishock]

                    # ii. next-period states
                    p_plus = trans.p_plus_func(sim.p[t,i],psi,par)
                    n_plus = trans.n_plus_func(sim.d[t,i],par)
                    m_plus = trans.m_plus_func(sim.a[t,i],p_plus,xi,par)
                    x_plus = trans.x_plus_func(m_plus,n_plus,par)

                    # iii. weight
                    weight = psi_w*xi_w

                    # iv. next-period choices
                    optimal_choice(t+1,p_plus,n_plus,m_plus,x_plus,discrete_plus,d_plus,c_plus,a_plus,sol,par)

                    # v. next-period marginal utility
                    RHS += weight*par.beta*par.R*utility.marg_func(c_plus[0],d_plus[0],par)
            
                euler_error[t,i] = sim.c[t,i] - utility.inv_marg_func(RHS,sim.d[t,i],par)
                euler_error_c[t,i] = sim.c[t,i]

@njit(parallel=True)
def calc_utility(sim,sol,par):
    """ calculate utility for each individual """

    # unpack
    u = sim.utility
    
    for t in range(par.simT):
        for i in prange(par.simN):

            u[i] += par.beta**t*utility.func(sim.c[t,i],sim.d[t,i],par)
            
