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
    c = sol.c_keep[t]

    for ip in prange(par.Np):
        
        # temporary containers
        m_temp = np.zeros(par.Na)
        c_temp = np.zeros(par.Na)
        v_temp = np.zeros(par.Na)

        for idb in range(par.Ndb):
            
            # use euler equation
            db = par.grid_db[idb]
            for ia in range(par.Na):
                c_temp[ia] = utility.inv_marg_func(sol.q[t,ip,idb,ia],db,par)
                m_temp[ia] = par.grid_a[ia] + c_temp[ia]
        
            # upperenvelope
            use_inv_w = True
            negm_upperenvelope(par.grid_a,m_temp,c_temp,sol.inv_w[t,ip,idb,:],
                par.grid_m,c[ip,idb,:],v_temp,db,par)
            
            # negative inverse
            for ia in range(par.Na):
                inv_v[ip,idb,ia] = -1/v_temp[ia]