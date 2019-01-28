import numpy as np
from numba import njit, prange

 # consav
from consav import linear_interp # for linear interpolation

# local modules
import utility

@njit(parallel=True)
def solve_bellman(t,sol,par):
    """solve the bellman equation using the endogenous grid method"""

    # unpack
    c = sol.c[t]

    for ip in prange(par.Np):
        
        # a. temporary container
        m_temp = np.zeros(par.Na+1)
        c_temp = np.zeros(par.Na+1)

        # b. invert Euler equation
        for ia in range(par.Na):
            c_temp[ia+1] = utility.inv_marg_func(sol.q[ip,ia],par)
            m_temp[ia+1] = par.grid_a[ia] + c_temp[ia+1]
        
        # b. re-interpolate consumption to common grid
        if par.do_simple_w: # use an explicit loop
            for im in range(par.Nm):
                c[ip,im] = linear_interp.interp_1d(m_temp,c_temp,par.grid_m[im])
        else: # use a vectorized call
            linear_interp.interp_1d_vec_mon_noprep(m_temp,c_temp,par.grid_m,c[ip,:])

