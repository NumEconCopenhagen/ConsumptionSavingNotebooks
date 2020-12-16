import numpy as np
from numba import njit

# consav
from consav import linear_interp # for linear interpolation

@njit
def compute(t,sol,par,G2EGM=True):

    # unpack
    w = sol.w[t]
    wa = sol.wa[t]
    if G2EGM:
        wb = sol.wb[t]

    # loop over outermost post-decision state
    for i_b in range(par.Nb_pd):

        # a. initialize
        w[i_b,:] = 0
        wa[i_b,:] = 0
        if G2EGM:
            wb[i_b,:] = 0

        # b. working memoery
        inv_v_plus = np.zeros(par.Na_pd)
        inv_vm_plus = np.zeros(par.Na_pd)
        if G2EGM:
            inv_vn_plus = np.zeros(par.Na_pd)
        
        inv_v_ret_plus = np.zeros(par.Na_pd)
        inv_vm_ret_plus = np.zeros(par.Na_pd)
        if G2EGM:
            inv_vn_ret_plus = np.zeros(par.Na_pd)

        # c. loop over shocks
        for i_eta in range(par.Neta):
            
            # i. next period states
            m_plus = par.Ra*par.grid_a_pd + par.eta[i_eta]
            n_plus = par.Rb*par.grid_b_pd[i_b]
            m_plus_ret = m_plus + n_plus

            # ii. prepare interpolation in p direction
            prep = linear_interp.interp_2d_prep(par.grid_n,n_plus,par.Na_pd)
            prep_ret = linear_interp.interp_1d_prep(par.Na_pd)

            # iii. interpolations

            # work
            linear_interp.interp_2d_only_last_vec_mon(prep,par.grid_n,par.grid_m,sol.inv_v[t+1],n_plus,m_plus,inv_v_plus)
            linear_interp.interp_2d_only_last_vec_mon_rep(prep,par.grid_n,par.grid_m,sol.inv_vm[t+1],n_plus,m_plus,inv_vm_plus)
            if G2EGM:
                linear_interp.interp_2d_only_last_vec_mon_rep(prep,par.grid_n,par.grid_m,sol.inv_vn[t+1],n_plus,m_plus,inv_vn_plus)

            # retire
            linear_interp.interp_1d_vec_mon(prep_ret,sol.m_ret[t+1],sol.inv_v_ret[t+1],m_plus_ret,inv_v_ret_plus)
            linear_interp.interp_1d_vec_mon_rep(prep_ret,sol.m_ret[t+1],sol.inv_vm_ret[t+1],m_plus_ret,inv_vm_ret_plus)
            if G2EGM:
                linear_interp.interp_1d_vec_mon_rep(prep_ret,sol.m_ret[t+1],sol.inv_vn_ret[t+1],m_plus_ret,inv_vn_ret_plus)

            # iv. accumulate
            for i_a in range(par.Na_pd):

                if inv_v_ret_plus[i_a] > inv_v_plus[i_a]:
                    w_now = -1.0/inv_v_ret_plus[i_a]
                    wa_now = 1.0/inv_vm_ret_plus[i_a]
                    if G2EGM:
                        wb_now = 1.0/inv_vn_ret_plus[i_a]
                else:
                    w_now = -1.0/inv_v_plus[i_a]
                    wa_now = 1.0/inv_vm_plus[i_a]
                    if G2EGM:
                        wb_now = 1.0/inv_vn_plus[i_a]
                
                w[i_b,i_a] += par.w_eta[i_eta]*par.beta*w_now
                wa[i_b,i_a] += par.w_eta[i_eta]*par.Ra*par.beta*wa_now
                if G2EGM:
                    wb[i_b,i_a] += par.w_eta[i_eta]*par.Rb*par.beta*wb_now