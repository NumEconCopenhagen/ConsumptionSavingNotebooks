import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp # for linear interpolation

# local modules
import utility
import trans

@njit(parallel=True)
def compute_wq(t,sol,par,compute_q=False):
    """ compute the post-decision functions w and/or q """

    # unpack
    inv_w = sol.inv_w[t]
    q = sol.q[t]

    # loop over outermost post-decision state
    for ip in prange(par.Np):

        # allocate temporary containers
        m_plus = np.zeros(par.Na) # container, same lenght as grid_a
        x_plus = np.zeros(par.Na)
        w = np.zeros(par.Na) 
        inv_v_keep_plus = np.zeros(par.Na)
        c_keep_plus = np.zeros(par.Na)
        inv_v_adj_plus = np.zeros(par.Na)
        d_adj_plus = np.zeros(par.Na)
        c_adj_plus = np.zeros(par.Na)
        
        # loop over other outer post-decision states
        for idb in range(par.Ndb):

            # a. permanent income and durable stock
            p = par.grid_p[ip]
            db = par.grid_db[idb]

            # b. initialize at zero
            for ia in range(par.Na):
                w[ia] = 0.0
                q[ip,idb,ia] = 0.0

            # c. loop over shocks and then end-of-period assets
            for ishock in range(par.Nshocks):
                
                # i. shocks
                psi_plus = par.psi[ishock]
                psi_plus_w = par.psi_w[ishock]
                xi_plus = par.xi[ishock]
                xi_plus_w = par.xi_w[ishock]

                # ii. next-period income and durables
                p_plus = trans.p_plus_func(p,psi_plus,par)
                db_plus = trans.db_plus_func(db,par)

                # iii. prepare interpolators
                prep_keep = linear_interp.interp_3d_prep(par.grid_p,par.grid_db,p_plus,db_plus,par.Na)
                prep_adj = linear_interp.interp_2d_prep(par.grid_p,p_plus,par.Na)

                # iv. weight
                weight = psi_plus_w*xi_plus_w

                # v. next-period cash-on-hand and total resources
                for ia in range(par.Na):
        
                    m_plus[ia] = trans.m_plus_func(par.grid_a[ia],p_plus,xi_plus,par)
                    x_plus[ia] = trans.x_plus_func(m_plus[ia],db_plus,par)
                
                # vi. interpolate
                linear_interp.interp_3d_only_last_vec_mon(prep_keep,par.grid_p,par.grid_db,par.grid_m,sol.inv_v_keep[t+1],p_plus,db_plus,m_plus,inv_v_keep_plus)
                linear_interp.interp_2d_only_last_vec_mon(prep_adj,par.grid_p,par.grid_x,sol.inv_v_adj[t+1],p_plus,x_plus,inv_v_adj_plus)
                if compute_q:
                    linear_interp.interp_3d_only_last_vec_mon_rep(prep_keep,par.grid_p,par.grid_db,par.grid_m,sol.c_keep[t+1],p_plus,db_plus,m_plus,c_keep_plus)
                    linear_interp.interp_2d_only_last_vec_mon_rep(prep_adj,par.grid_p,par.grid_x,sol.d_adj[t+1],p_plus,x_plus,d_adj_plus)
                    linear_interp.interp_2d_only_last_vec_mon_rep(prep_adj,par.grid_p,par.grid_x,sol.c_adj[t+1],p_plus,x_plus,c_adj_plus)
                
                # vii. max and accumulate
                if compute_q:

                    for ia in range(par.Na):                                

                        keep = inv_v_keep_plus[ia] > inv_v_adj_plus[ia]
                        if keep:
                            v_plus = -1/inv_v_keep_plus[ia]
                            d_plus = db_plus
                            c_plus = c_keep_plus[ia]
                        else:
                            v_plus = -1/inv_v_adj_plus[ia]
                            d_plus = d_adj_plus[ia]
                            c_plus = c_adj_plus[ia]

                        w[ia] += weight*par.beta*v_plus
                        q[ip,idb,ia] += weight*par.beta*par.R*utility.marg_func(c_plus,d_plus,par)

                else:

                    for ia in range(par.Na):
                        w[ia] += weight*par.beta*(-1.0/np.fmax(inv_v_keep_plus[ia],inv_v_adj_plus[ia]))
        
            # d. transform post decision value function
            for ia in range(par.Na):
                inv_w[ip,idb,ia] = -1/w[ia]
