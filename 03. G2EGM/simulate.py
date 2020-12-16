import numpy as np
from numba import njit

# consav
from consav import linear_interp # for linear interpolation

# local modules
import pens
import utility

@njit
def euler(sim,sol,par):
    
    euler = sim.euler

    # a. grids
    min_m = 0.50
    min_n = 0.01    
     
    m_max = 5.00
    n_max = 5.00
     
    n_grid = np.linspace(min_n,n_max,par.eulerK)
    m_grid = np.linspace(min_m,m_max,par.eulerK)

    # b. loop over time
    for t in range(par.T-1):
        for i_n in range(par.eulerK):
            for i_m in range(par.eulerK):

                    # i. states
                    n = n_grid[i_n]
                    m = m_grid[i_m]
                    m_retire = m_grid[i_m]+n_grid[i_n]

                    # ii. discrete choice
                    inv_v_retire = linear_interp.interp_1d(sol.m_ret[t],sol.inv_v_ret[t],m_retire)
                    inv_v = linear_interp.interp_2d(par.grid_n,par.grid_m,sol.inv_v[t],n,m)

                    if inv_v_retire > inv_v: continue
                    
                    # iii. continuous choice
                    c = np.fmin(linear_interp.interp_2d(par.grid_n,par.grid_m,sol.c[t],n,m),m)
                    d = np.fmax(linear_interp.interp_2d(par.grid_n,par.grid_m,sol.d[t],n,m),0)
                    a = m-c-d
                    b = n+d+pens.func(d,par)

                    if a < 0.001: continue

                    # iv. shocks
                    RHS = 0
                    for i_eta in range(par.Neta):
             
                        # o. state variables
                        n_plus = par.Rb*b
                        m_plus = par.Ra*a + par.eta[i_eta]
                        m_retire_plus = m_plus + n_plus

                        # oo. discrete choice
                        inv_v_retire = linear_interp.interp_1d(sol.m_ret[t+1],sol.inv_v_ret[t+1],m_retire_plus)
                        inv_v = linear_interp.interp_2d(par.grid_n,par.grid_m,sol.inv_v[t+1],n_plus,m_plus)

                        # ooo. continous choice
                        if inv_v_retire > inv_v:
                            c_plus = np.fmin(linear_interp.interp_1d(sol.m_ret[t+1],sol.c_ret[t+1],m_retire_plus),m_retire_plus)
                        else:
                            c_plus = np.fmin(linear_interp.interp_2d(par.grid_n,par.grid_m,sol.c[t+1],n_plus,m_plus),m_plus)

                        # oooo. accumulate
                        RHS += par.w_eta[i_eta]*par.beta*par.Ra*utility.marg_func(c_plus,par)    

                    # v. euler error
                    euler_raw = c - utility.inv_marg_func(RHS,par)
                    euler[t,i_m,i_n] = np.log10(np.abs(euler_raw/c)+1e-16)