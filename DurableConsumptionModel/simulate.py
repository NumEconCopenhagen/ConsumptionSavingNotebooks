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
    n1 = sim.n1
    n2 = sim.n2
    m = sim.m
    c = sim.c
    d = sim.d
    d1 = sim.d1
    d2 = sim.d2
    a = sim.a
    discrete = sim.discrete
    
    for t in range(par.T):
        for i in prange(par.simN):
            
            # a. beginning of period states
            if t == 0:
                p[t,i] = trans.p_plus_func(sim.p0[i],sim.psi[t,i],par)
                if par.do_2d:
                    n1[t,i] = trans.n1_plus_func(sim.d10[i],par)
                    n2[t,i] = trans.n2_plus_func(sim.d20[i],par)
                else:
                    n[t,i] = trans.n_plus_func(sim.d0[i],par)
                m[t,i] = trans.m_plus_func(sim.a0[i],p[t,i],sim.xi[t,i],par)
            else:
                p[t,i] = trans.p_plus_func(p[t-1,i],sim.psi[t,i],par)
                if par.do_2d:
                    n1[t,i] = trans.n1_plus_func(d1[t-1,i],par)
                    n2[t,i] = trans.n2_plus_func(d2[t-1,i],par)
                else:
                    n[t,i] = trans.n_plus_func(d[t-1,i],par)
                m[t,i] = trans.m_plus_func(a[t-1,i],p[t,i],sim.xi[t,i],par)
            
            # b. optimal choices and post decision states
            if par.do_2d:
                optimal_choice_2d(t,p[t,i],n1[t,i],n2[t,i],m[t,i],discrete[t,i:],d1[t,i:],d2[t,i:],c[t,i:],a[t,i:],sol,par)
            else:
                optimal_choice(t,p[t,i],n[t,i],m[t,i],discrete[t,i:],d[t,i:],c[t,i:],a[t,i:],sol,par)
            
@njit            
def optimal_choice(t,p,n,m,discrete,d,c,a,sol,par):

    x = trans.x_plus_func(m,n,par)

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
def optimal_choice_2d(t,p,n1,n2,m,discrete,d1,d2,c,a,sol,par):

    # a. discrete choice
    inv_v = 0
    inv_v_keep = linear_interp.interp_4d(par.grid_p,par.grid_n,par.grid_n,par.grid_m,sol.inv_v_keep_2d[t],p,n1,n2,m)

    x_full = m + (1-par.tau1)*n1 + (1-par.tau2)*n2
    inv_v_adj_full = linear_interp.interp_2d(par.grid_p,par.grid_x,sol.inv_v_adj_full_2d[t],p,x_full)
    
    x_d1 = m + (1-par.tau1)*n1
    inv_v_adj_d1 = linear_interp.interp_3d(par.grid_p,par.grid_n,par.grid_x,sol.inv_v_adj_d1_2d[t],p,n2,x_d1)

    x_d2 = m + (1-par.tau2)*n2
    inv_v_adj_d2 = linear_interp.interp_3d(par.grid_p,par.grid_n,par.grid_x,sol.inv_v_adj_d2_2d[t],p,n1,x_d2)

    keep = False
    adj_full = False
    adj_d1 = False
    adj_d2 = False

    if inv_v_keep > inv_v:
        inv_v = inv_v_keep
        keep = True

    if inv_v_adj_full > inv_v:
        inv_v = inv_v_adj_full
        keep = False
        adj_full = True

    if inv_v_adj_d1 > inv_v:
        inv_v = inv_v_adj_d1
        keep = False
        adj_full = False
        adj_d1 = True

    if inv_v_adj_d2 > inv_v:
        inv_v = inv_v_adj_d2
        keep = False
        adj_full = False
        adj_d1 = False
        adj_d2 = True

    # b. continuous choices
    if keep: 
            
        discrete[0] = 0

        d1[0] = n1
        d2[0] = n2

        c[0] = linear_interp.interp_4d(
            par.grid_p,par.grid_n,par.grid_n,par.grid_m,sol.c_keep_2d[t],
            p,n1,n2,m)

        if c[0] > m: 
            c[0] = m
            a[0] = 0.0
        else:
            a[0] = m - c[0]

    elif adj_full:

        discrete[0] = 1
        
        d1[0] = linear_interp.interp_2d(
            par.grid_p,par.grid_x,sol.d1_adj_full_2d[t],
            p,x_full)

        d2[0] = linear_interp.interp_2d(
            par.grid_p,par.grid_x,sol.d2_adj_full_2d[t],
            p,x_full)            

        c[0] = linear_interp.interp_2d(
            par.grid_p,par.grid_x,sol.c_adj_full_2d[t],
            p,x_full)

        tot = d1[0]+d2[0]+c[0]
        if tot > x_full: 
            d1[0] *= x_full/tot
            d2[0] *= x_full/tot
            c[0] *= x_full/tot
            a[0] = 0.0
        else:
            a[0] = x_full - tot
            
    elif adj_d1:

        discrete[0] = 2
        
        d1[0] = linear_interp.interp_3d(
            par.grid_p,par.grid_n,par.grid_x,sol.d1_adj_d1_2d[t],
            p,n2,x_d1)

        d2[0] = n2           

        c[0] = linear_interp.interp_3d(
            par.grid_p,par.grid_n,par.grid_x,sol.c_adj_d1_2d[t],
            p,n2,x_d1)

        tot = d1[0]+c[0]
        if tot > x_d1: 
            d1[0] *= x_d1/tot
            c[0] *= x_d1/tot
            a[0] = 0.0
        else:
            a[0] = x_d1 - tot

    elif adj_d2:

        discrete[0] = 3
        
        d1[0] = n1

        d2[0] = linear_interp.interp_3d(
            par.grid_p,par.grid_n,par.grid_x,sol.d2_adj_d2_2d[t],
            p,n1,x_d2)

        c[0] = linear_interp.interp_3d(
            par.grid_p,par.grid_n,par.grid_x,sol.c_adj_d2_2d[t],
            p,n1,x_d2)

        tot = d2[0]+c[0]
        if tot > x_d2: 
            d2[0] *= x_d2/tot
            c[0] *= x_d2/tot
            a[0] = 0.0
        else:
            a[0] = x_d2 - tot            

@njit            
def euler_errors(sim,sol,par):

    # unpack
    euler_error = sim.euler_error
    euler_error_c = sim.euler_error_c
    
    for i in prange(par.simN):
        
        discrete_plus = np.zeros(1)
        d_plus = np.zeros(1)
        d1_plus = np.zeros(1)
        d2_plus = np.zeros(1)
        c_plus = np.zeros(1)
        a_plus = np.zeros(1)

        for t in range(par.T-1):

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
                    if par.do_2d:
                        n1_plus = trans.n1_plus_func(sim.d1[t,i],par)
                        n2_plus = trans.n2_plus_func(sim.d2[t,i],par)
                    else:
                        n_plus = trans.n_plus_func(sim.d[t,i],par)
                    m_plus = trans.m_plus_func(sim.a[t,i],p_plus,xi,par)

                    # iii. weight
                    weight = psi_w*xi_w

                    # iv. next-period choices
                    if par.do_2d:
                        optimal_choice_2d(t+1,p_plus,n1_plus,n2_plus,m_plus,discrete_plus,d1_plus,d2_plus,c_plus,a_plus,sol,par)
                    else:
                        optimal_choice(t+1,p_plus,n_plus,m_plus,discrete_plus,d_plus,c_plus,a_plus,sol,par)

                    # v. next-period marginal utility
                    if par.do_2d:
                        RHS += weight*par.beta*par.R*utility.marg_func_2d(c_plus[0],d1_plus[0],d2_plus[0],par)
                    else:
                        RHS += weight*par.beta*par.R*utility.marg_func(c_plus[0],d_plus[0],par)
                
                if par.do_2d:
                    euler_error[t,i] = sim.c[t,i] - utility.inv_marg_func_2d(RHS,sim.d1[t,i],sim.d2[t,i],par)
                else:
                    euler_error[t,i] = sim.c[t,i] - utility.inv_marg_func(RHS,sim.d[t,i],par)

                euler_error_c[t,i] = sim.c[t,i]

@njit(parallel=True)
def calc_utility(sim,sol,par):
    """ calculate utility for each individual """

    # unpack
    u = sim.utility
    
    for t in range(par.T):
        for i in prange(par.simN):
            
            if par.do_2d:
                u[i] += par.beta**t*utility.func_2d(sim.c[t,i],sim.d1[t,i],sim.d2[t,i],par)
            else:
                u[i] += par.beta**t*utility.func(sim.c[t,i],sim.d[t,i],par)
            
