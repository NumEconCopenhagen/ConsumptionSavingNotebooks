import numpy as np
from numba import njit, prange

 # consav
from consav import linear_interp # for linear interpolation
from consav import golden_section_search # for optimization in keeper problem

# quantecon
import quantecon as qe # for optimization in adjuster problem

# local modules
import utility
import trans

@njit
def value_of_choice(t,c,d,p,x,inv_v_keep,inv_v_adj,par):
    
    # a. end-of-period-assets
    a = x-c-d
    
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
        n_plus = trans.n_plus_func(d,par)
        m_plus = trans.m_plus_func(a,p_plus,xi,par)
        x_plus = trans.x_plus_func(m_plus,n_plus,par)
        
        # iii. weight
        weight = psi_w*xi_w

        # iv. update
        inv_v_plus_keep_now = linear_interp.interp_3d(par.grid_p,par.grid_n,par.grid_m,
                                                      inv_v_keep[t+1], p_plus, n_plus,
                                                      m_plus)

        inv_v_plus_adj_now = linear_interp.interp_2d(par.grid_p, par.grid_x, 
                                                     inv_v_adj[t+1], p_plus, x_plus)
        
        v_plus_now = - np.inf  # huge negative value
        
        if inv_v_plus_keep_now > inv_v_plus_adj_now and inv_v_plus_keep_now > 0:
            v_plus_now = -1/inv_v_plus_keep_now
        elif inv_v_plus_adj_now > 0:
            v_plus_now = -1/inv_v_plus_adj_now
            
        w += weight*par.beta*v_plus_now
        
    # v. total value
    v = utility.func(c,d,par) + w

    return v # we are minimizing
         
########
# keep #
########
    
@njit
def obj_keep(c,t,n,p,m,inv_v_keep,inv_v_adj,par): # max wrt. c
    """ evaluate bellman equation keepers """
    
    # unpack (helps numba optimize)
    d = n
    x = m + n

    penalty = 0
    if c <= 1e-12:
        penalty = 10_000*(1e-12-c)
        c = 1e-12

    return -value_of_choice(t,c,d,p,x,inv_v_keep,inv_v_adj,par) + penalty # minimization

# b. solve keep
@njit(parallel=True)
def solve_keep(t,sol,par):
    """solve bellman equation for keepers using vfi"""

    # unpack (helps numba optimize)
    inv_v = sol.inv_v_keep[t]
    c = sol.c_keep[t]
    
    # keep: loop over outer states
    for i_p in prange(par.Np): # loop in parallel        
        for i_n in range(par.Nn):

            p = par.grid_p[i_p]
            n = par.grid_n[i_n]
            
            # loop over cash-on-hand
            for i_m in range(par.Nm):

                # a. cash-on-hand
                m = par.grid_m[i_m]
                if i_m == 0: 
                    c[i_p,i_n,i_m] = 0
                    inv_v[i_p,i_n,i_m] = 0
                    continue
                
                # b. optimal choice
                c_low = np.fmin(m/2,1e-8)
                c_high = m
                c_opt = golden_section_search.optimizer(obj_keep,c_low,c_high,
                    args=(t,n,p,m,sol.inv_v_keep,sol.inv_v_adj,par),tol=par.tol)
                c[i_p,i_n,i_m] = c_opt

                # c. optimal value
                v = -obj_keep(c[i_p,i_n,i_m],t,n,p,m,sol.inv_v_keep,sol.inv_v_adj,par)
                inv_v[i_p,i_n,i_m] = -1/v
                
#######
# adj #
#######

@njit
def obj_adj(choices,t,p,x,inv_v_keep,inv_v_adj,par):
    """ evaluate bellman equation adjusters """
    
    # load consumption and durable consumption guesses from optimizer
    c = choices[0]
    d = choices[1]

    # create penalty function to comply with c + d < x (not included in bounds of Nelder-Mead)
    penalty = 0

    if c+d > x:

        penalty = 10_000*(c+d-x)
        c /= (c+d)/x
        d /= (c+d)/x

    return value_of_choice(t,c,d,p,x,inv_v_keep,inv_v_adj,par) - penalty # maximization

# c. solve adjuster problem (2D optimization)
@njit(parallel=True)
def solve_adj(t,sol,par):

    # unpack (helps numba optimize)
    inv_v = sol.inv_v_adj[t]
    d = sol.d_adj[t]
    c = sol.c_adj[t]

    # adj: loop over outer states
    # loop over p state
    for i_p in prange(par.Np):
        
        p = par.grid_p[i_p]
        choices = np.zeros(2)
        
        # loop over x state
        for i_x in range(par.Nx):
                
                # a. cash-on-hand
                x = par.grid_x[i_x]
                
                if i_x == 0:
                    d[i_p,i_x] = 0
                    c[i_p,i_x] = 0
                    inv_v[i_p,i_x] = 0
                    continue
                elif i_x == 1: # initial guesses for Nelder-Mead
                    choices[0] = x/3
                    choices[1] = x/3
        
                # b. optimal choice
                results = qe.optimize.nelder_mead(obj_adj,choices, 
                    bounds = np.array([[1e-8,x],[1e-8,x]]), 
                    args=(t,p,x,sol.inv_v_keep,sol.inv_v_adj,par),
                    tol_x=par.tol, 
                    max_iter=1000)

                choices = results[0]
                minf = results[1]

                # c. optimal values
                c[i_p,i_x] = choices[0]
                d[i_p,i_x] = choices[1]  
                inv_v[i_p,i_x] = -1/minf # minus due to maximization