import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp # for linear interpolation
from consav import upperenvelope

# local modules
import utility

negm_upperenvelope = upperenvelope.create(utility.func,use_inv_w=True)

@njit(parallel=True)
def solve_keep(t,sol,par):
    """solve the bellman equation using the endogenous grid method"""

    # unpack
    inv_v = sol.inv_v_keep[t]
    inv_marg_u = sol.inv_marg_u_keep[t]
    c = sol.c_keep[t]
    q_c = sol.q_c[t]
    q_m = sol.q_m[t]

    for i_p in prange(par.Np):
        
        # temporary container
        v_ast_vec = np.zeros(par.Nm)

        for i_n in range(par.Nn):
            
            # use euler equation
            n = par.grid_n[i_n]
            for i_a in range(par.Na):
                q_c[i_p,i_n,i_a] = utility.inv_marg_func(sol.q[t,i_p,i_n,i_a],n,par)
                q_m[i_p,i_n,i_a] = par.grid_a[i_a] + q_c[i_p,i_n,i_a]
        
            # upperenvelope
            negm_upperenvelope(par.grid_a,q_m[i_p,i_n],q_c[i_p,i_n],sol.inv_w[t,i_p,i_n],
               par.grid_m,c[i_p,i_n],v_ast_vec,n,par)        

            # negative inverse
            for i_m in range(par.Nm):
                inv_v[i_p,i_n,i_m] = -1/v_ast_vec[i_m]
                if par.do_marg_u:
                    inv_marg_u[i_p,i_n,i_m] = 1/utility.marg_func(c[i_p,i_n,i_m],n,par)