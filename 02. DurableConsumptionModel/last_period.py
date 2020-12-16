import numpy as np
from numba import njit, prange

# consav
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

@njit
def obj_last_period_full_2d(d,x,par):
    """ objective function in last period """
    
    # note, use the assumption that gamma = 0.5 -> d1 = d2 = d

    # implied consumption (rest)
    c = x-2*d

    return -utility.func_2d(c,d,d,par)

@njit
def obj_last_period_d1_2d(d1,n2,x,par):
    """ objective function in last period """
    
    # implied consumption (rest)
    c = x-d1

    return -utility.func_2d(c,d1,n2,par)

@njit
def obj_last_period_d2_2d(d2,n1,x,par):
    """ objective function in last period """
    
    # implied consumption (rest)
    c = x-d2

    return -utility.func_2d(c,n1,d2,par)

@njit(parallel=True)
def solve(t,sol,par):
    """ solve the problem in the last period """

    # unpack
    inv_v_keep = sol.inv_v_keep[t]
    inv_marg_u_keep = sol.inv_marg_u_keep[t]
    c_keep = sol.c_keep[t]
    inv_v_adj = sol.inv_v_adj[t]
    inv_marg_u_adj = sol.inv_marg_u_adj[t]
    d_adj = sol.d_adj[t]
    c_adj = sol.c_adj[t]

    # a. keep
    for i_p in prange(par.Np):
        for i_n in range(par.Nn):
            for i_m in range(par.Nm):
                            
                # i. states
                n = par.grid_n[i_n]
                m = par.grid_m[i_m]

                if m == 0: # forced c = 0 
                    c_keep[i_p,i_n,i_m] = 0
                    inv_v_keep[i_p,i_n,i_m] = 0
                    inv_marg_u_keep[i_p,i_n,i_m] = 0
                    continue
                
                # ii. optimal choice
                c_keep[i_p,i_n,i_m] = m

                # iii. optimal value
                v_keep = utility.func(c_keep[i_p,i_n,i_m],n,par)
                inv_v_keep[i_p,i_n,i_m] = -1.0/v_keep
                inv_marg_u_keep[i_p,i_n,i_m] = 1.0/utility.marg_func(c_keep[i_p,i_n,i_m],n,par)

    # b. adj
    for i_p in prange(par.Np):
        for i_x in range(par.Nx):
            
            # i. states
            x = par.grid_x[i_x]

            if x == 0: # forced c = d = 0
                d_adj[i_p,i_x] = 0
                c_adj[i_p,i_x] = 0
                inv_v_adj[i_p,i_x] = 0
                inv_marg_u_adj[i_p,i_x] = 0
                continue

            # ii. optimal choices
            d_low = np.fmin(x/2,1e-8)
            d_high = np.fmin(x,par.n_max)            
            d_adj[i_p,i_x] = golden_section_search.optimizer(obj_last_period,d_low,d_high,args=(x,par),tol=par.tol)
            c_adj[i_p,i_x] = x-d_adj[i_p,i_x]

            # iii. optimal value
            v_adj = -obj_last_period(d_adj[i_p,i_x],x,par)
            inv_v_adj[i_p,i_x] = -1.0/v_adj
            inv_marg_u_adj[i_p,i_x] = 1.0/utility.marg_func(c_adj[i_p,i_x],d_adj[i_p,i_x],par)


@njit(parallel=True)
def solve_2d(t,sol,par):
    """ solve the problem in the last period """

    # unpack
    inv_v_keep = sol.inv_v_keep_2d[t]
    inv_marg_u_keep = sol.inv_marg_u_keep_2d[t]
    c_keep = sol.c_keep_2d[t]

    inv_v_adj_full = sol.inv_v_adj_full_2d[t]
    inv_marg_u_adj_full = sol.inv_marg_u_adj_full_2d[t]
    d1_adj_full = sol.d1_adj_full_2d[t]
    d2_adj_full = sol.d2_adj_full_2d[t]
    c_adj_full = sol.c_adj_full_2d[t]

    inv_v_adj_d1 = sol.inv_v_adj_d1_2d[t]
    inv_marg_u_adj_d1 = sol.inv_marg_u_adj_d1_2d[t]
    d1_adj_d1 = sol.d1_adj_d1_2d[t]
    c_adj_d1 = sol.c_adj_d1_2d[t]

    inv_v_adj_d2 = sol.inv_v_adj_d2_2d[t]
    inv_marg_u_adj_d2 = sol.inv_marg_u_adj_d2_2d[t]
    d2_adj_d2 = sol.d2_adj_d2_2d[t]
    c_adj_d2 = sol.c_adj_d2_2d[t]

    # a. keep
    for i_p in prange(par.Np):

        for i_n1 in range(par.Nn):
            for i_n2 in range(par.Nn):
                for i_m in range(par.Nm):
                                
                    # i. states
                    n1 = par.grid_n[i_n1]
                    n2 = par.grid_n[i_n2]
                    m = par.grid_m[i_m]

                    if m == 0: # forced c = 0 
                        c_keep[i_p,i_n1,i_n2,i_m] = 0
                        inv_v_keep[i_p,i_n1,i_n2,i_m] = 0
                        inv_marg_u_keep[i_p,i_n1,i_n2,i_m] = 0
                        continue
                    
                    # ii. optimal choice
                    c_keep[i_p,i_n1,i_n2,i_m] = m

                    # iii. optimal value
                    v_keep = utility.func_2d(c_keep[i_p,i_n1,i_n2,i_m],n1,n2,par)
                    inv_v_keep[i_p,i_n1,i_n2,i_m] = -1.0/v_keep
                    inv_marg_u_keep[i_p,i_n1,i_n2,i_m] = 1.0/utility.marg_func_2d(c_keep[i_p,i_n1,i_n2,i_m],n1,n2,par)

    # b. adj full (use gamma = 0.5)
    for i_p in prange(par.Np):
        for i_x in range(par.Nx):
            
            # i. states
            x = par.grid_x[i_x]

            if x == 0: # forced c = d1 = d2 = 0
                d1_adj_full[i_p,i_x] = 0
                d2_adj_full[i_p,i_x] = 0
                c_adj_full[i_p,i_x] = 0
                inv_v_adj_full[i_p,i_x] = 0
                inv_marg_u_adj_full[i_p,i_x] = 0
                continue

            # ii. optimal choices
            d_low = np.fmin(x/2,1e-8)
            d_high = np.fmin(x/2,par.n_max)            
            d1_adj_full[i_p,i_x] = golden_section_search.optimizer(obj_last_period_full_2d,d_low,d_high,args=(x,par),tol=par.tol)
            d2_adj_full[i_p,i_x] = d1_adj_full[i_p,i_x]
            c_adj_full[i_p,i_x] = x-2*d1_adj_full[i_p,i_x]

            # iii. optimal value
            v_adj = -obj_last_period_full_2d(d1_adj_full[i_p,i_x],x,par)
            inv_v_adj_full[i_p,i_x] = -1.0/v_adj
            inv_marg_u_adj_full[i_p,i_x] = 1.0/utility.marg_func_2d(c_adj_full[i_p,i_x],d1_adj_full[i_p,i_x],d2_adj_full[i_p,i_x],par)

    # c. adj d1
    for i_p in prange(par.Np):

        for i_n2 in range(par.Nn):
            for i_x in range(par.Nx):
            
                # i. states
                n2 = par.grid_n[i_n2]
                x = par.grid_x[i_x]

                if x == 0: # forced c = d1 = 0
                    d1_adj_d1[i_p,i_n2,i_x] = 0
                    c_adj_d1[i_p,i_n2,i_x] = 0
                    inv_v_adj_d1[i_p,i_n2,i_x] = 0
                    inv_marg_u_adj_d1[i_p,i_n2,i_x] = 0
                    continue

                # ii. optimal choices
                d_low = np.fmin(x/2,1e-8)
                d_high = np.fmin(x,par.n_max)            
                d1_adj_d1[i_p,i_n2,i_x] = golden_section_search.optimizer(obj_last_period_d1_2d,d_low,d_high,args=(n2,x,par),tol=par.tol)
                c_adj_d1[i_p,i_n2,i_x] = x-d1_adj_d1[i_p,i_n2,i_x]

                # iii. optimal value
                v_adj = -obj_last_period_d1_2d(d1_adj_d1[i_p,i_n2,i_x],n2,x,par)
                inv_v_adj_d1[i_p,i_n2,i_x] = -1.0/v_adj
                inv_marg_u_adj_d1[i_p,i_n2,i_x] = 1.0/utility.marg_func_2d(c_adj_d1[i_p,i_n2,i_x],d1_adj_d1[i_p,i_n2,i_x],n2,par)            

    # d. adj d2
    for i_p in prange(par.Np):
        for i_n1 in range(par.Nn):
            for i_x in range(par.Nx):
            
                # i. states
                n1 = par.grid_n[i_n1]
                x = par.grid_x[i_x]

                if x == 0: # forced c = d2 = 0
                    d2_adj_d2[i_p,i_n1,i_x] = 0
                    c_adj_d2[i_p,i_n1,i_x] = 0
                    inv_v_adj_d2[i_p,i_n1,i_x] = 0
                    inv_marg_u_adj_d2[i_p,i_n1,i_x] = 0
                    continue

                # ii. optimal choices
                d_low = np.fmin(x/2,1e-8)
                d_high = np.fmin(x,par.n_max)            
                d2_adj_d2[i_p,i_n1,i_x] = golden_section_search.optimizer(obj_last_period_d2_2d,d_low,d_high,args=(n1,x,par),tol=par.tol)
                c_adj_d2[i_p,i_n1,i_x] = x-d2_adj_d2[i_p,i_n1,i_x]

                # iii. optimal value
                v_adj = -obj_last_period_d2_2d(d2_adj_d2[i_p,i_n1,i_x],n1,x,par)
                inv_v_adj_d2[i_p,i_n1,i_x] = -1.0/v_adj
                inv_marg_u_adj_d2[i_p,i_n1,i_x] = 1.0/utility.marg_func_2d(c_adj_d2[i_p,i_n1,i_x],n1,d2_adj_d2[i_p,i_n1,i_x],par)                            