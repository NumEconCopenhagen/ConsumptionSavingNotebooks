import numpy as np
from numba import njit

import utility

# consav
from consav import linear_interp # for linear interpolation
from consav.grids import nonlinspace_jit # grids

@njit
def solve(sol,par,G2EGM=True):

    # a. last_period
    t = par.T-1

    sol.m_ret[t,:] = par.grid_m_ret
    sol.c_ret[t,:] = sol.m_ret[t,:]
    
    v = utility.func_ret(sol.c_ret[t,:],par)
    sol.inv_v_ret[t,:] = -1.0/v
    
    vm = utility.marg_func(sol.c_ret[t,:],par)
    sol.inv_vm_ret[t,:] = 1.0/vm
    if G2EGM:
        sol.inv_vn_ret[t,:] = sol.inv_vm_ret[t,:]

    # b. backwards inducation
    for j in range(2,par.T+1):
        
        t = par.T-j

        # i. optimal c choice
        m_plus = par.Ra*par.grid_a_ret + par.yret
        c_plus = np.zeros(m_plus.shape)
        linear_interp.interp_1d_vec(sol.m_ret[t+1,:],sol.c_ret[t+1,:],m_plus,c_plus)
        
        vm_plus = utility.marg_func(c_plus,par) 
        q = par.beta*par.Ra*vm_plus
        sol.c_ret[t,par.Nmcon_ret:] = utility.inv_marg_func(q,par)

        # ii. constraint            
        sol.c_ret[t,:par.Nmcon_ret] = nonlinspace_jit(1e-6,sol.c_ret[t,par.Nmcon_ret]*0.999,par.Nmcon_ret,par.phi_m)
    
        # iii. end-of-period assets and value-of-choice
        sol.a_ret[t,par.Nmcon_ret:] = par.grid_a_ret
        
        inv_v_plus = np.zeros(m_plus.shape)
        linear_interp.interp_1d_vec(sol.m_ret[t+1,:],sol.inv_v_ret[t+1,:],m_plus,inv_v_plus)
        v_plus = -1.0/inv_v_plus

        v1 = utility.func_ret(sol.c_ret[t,:par.Nmcon_ret],par) + par.beta*v_plus[0] 
        v2 = utility.func_ret(sol.c_ret[t,par.Nmcon_ret:],par) + par.beta*v_plus

        sol.inv_v_ret[t,:par.Nmcon_ret] = -1.0/v1
        sol.inv_v_ret[t,par.Nmcon_ret:] = -1.0/v2
                
        # iv. endogenous grid
        sol.m_ret[t,:] = sol.a_ret[t,:] + sol.c_ret[t,:]

        # v. marginal v
        vm = utility.marg_func(sol.c_ret[t,:],par)
        sol.inv_vm_ret[t,:] = 1.0/vm
        if G2EGM:
            sol.inv_vn_ret[t,:] = sol.inv_vm_ret[t,:]