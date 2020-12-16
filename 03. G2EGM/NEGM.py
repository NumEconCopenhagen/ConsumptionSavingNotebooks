import numpy as np
from numba import njit

# consav
from consav import linear_interp # for linear interpolation
from consav import upperenvelope
from consav import golden_section_search

# local modules
import utility
import pens

negm_upperenvelope = upperenvelope.create(utility.func,use_inv_w=False)

@njit
def solve_pure_c(t,sol,par):

    w = sol.w[t]
    wa = sol.wa[t]

    # unpack
    inv_v = sol.inv_v_pure_c[t]
    c = sol.c_pure_c[t]

    for i_b in range(par.Nb_pd):
        
        # temporary containers
        temp_c = np.zeros(par.Na_pd)
        temp_m = np.zeros(par.Na_pd)
        temp_v = np.zeros(par.Na_pd)
            
        # use euler equation
        for i_a in range(par.Na_pd):
            temp_c[i_a] = utility.inv_marg_func(wa[i_b,i_a],par)
            temp_m[i_a] = par.grid_a_pd[i_a] + temp_c[i_a]
    
        # upperenvelope
        negm_upperenvelope(par.grid_a_pd,temp_m,temp_c,w[i_b],
            par.grid_l,c[i_b,:],temp_v,par)        

        # negative inverse
        for i_m in range(par.Na_pd):
            inv_v[i_b,i_m] = -1/temp_v[i_m]

@njit
def obj_outer(d,n,m,t,sol,par):
    """ evaluate bellman equation """

    # a. cash-on-hand
    m_pure_c = m-d

    # b. durables
    n_pure_c = n + d + pens.func(d,par)
    
    # c. value-of-choice
    return -linear_interp.interp_2d(par.grid_b_pd,par.grid_l,sol.inv_v_pure_c[t],n_pure_c,m_pure_c)  # we are minimizing

@njit
def solve_outer(t,sol,par):

    # unpack output
    inv_v = sol.inv_v[t]
    inv_vm = sol.inv_vm[t]
    c = sol.c[t]
    d = sol.d[t]

    # loop over outer states
    for i_n in range(par.Nn):
            
        n = par.grid_n[i_n]

        # loop over m state
        for i_m in range(par.Nm):
            
            m = par.grid_m[i_m]
            
            # a. optimal choice
            d_low = 1e-8
            d_high = m-1e-8
            d[i_n,i_m] = golden_section_search.optimizer(obj_outer,d_low,d_high,args=(n,m,t,sol,par),tol=1e-8)

            # b. optimal value
            n_pure_c = n + d[i_n,i_m] + pens.func(d[i_n,i_m],par)
            m_pure_c = m - d[i_n,i_m]
            c[i_n,i_m] = np.fmin(linear_interp.interp_2d(par.grid_b_pd,par.grid_l,sol.c_pure_c[t],n_pure_c,m_pure_c),m_pure_c)
            inv_v[i_n,i_m] = -obj_outer(d[i_n,i_m],n,m,t,sol,par)

            # c. dcon
            obj_dcon = -obj_outer(0,n,m,t,sol,par)
            if obj_dcon > inv_v[i_n,i_m]:
                c[i_n,i_m] = linear_interp.interp_2d(par.grid_b_pd,par.grid_l,sol.c_pure_c[t],n,m)
                d[i_n,i_m] = 0
                inv_v[i_n,i_m] = obj_dcon

            # d. con
            w = linear_interp.interp_2d(par.grid_b_pd,par.grid_a_pd,sol.w[t],n,0)
            obj_con = -1.0/(utility.func(m,par) + w)
            if obj_con > inv_v[i_n,i_m]:
                c[i_n,i_m] = m
                d[i_n,i_m] = 0
                inv_v[i_n,i_m] = obj_con

            # e. derivative
            inv_vm[i_n,i_m] = 1.0/utility.marg_func(c[i_n,i_m],par)