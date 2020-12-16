import numpy as np
from numba import njit

# consav
from consav import linear_interp # for linear interpolation
from consav.grids import nonlinspace_jit # grids

# local modules
import pens
import utility
import upperenvelope

@njit
def inv_mn_and_v(c,d,a,b,w,par):

    v = utility.func(c,par) + w
    m = a+c+d
    n = b-d-pens.func(d,par)

    return m,n,v

@njit
def deviate_d_con(valt,n,c,a,w,par):
    
    for i_b in range(par.Nb_pd):
        for i_a in range(par.Na_pd):
        
            # a. choices
            d_x = par.delta_con*c[i_b,i_a]
            c_x = (1.0-par.delta_con)*c[i_b,i_a]
        
            # b. post-decision states            
            b_x = n[i_b,i_a] + d_x + pens.func(d_x,par)

            if not np.imag(b_x) == 0:
                valt[i_b,i_a] = np.nan
                continue
            
            # c. value
            w_x = linear_interp.interp_2d(par.grid_b_pd,par.grid_a_pd,w,b_x,a[i_b,i_a])
            v_x = utility.func(c_x,par) + w_x

            if not np.imag(v_x) == 0:
                valt[i_b,i_a] = np.nan
            else:
                valt[i_b,i_a] = v_x

@njit
def solve_ucon(out_c,out_d,out_v,w,wa,wb,par):

    num = 1

    # i. choices
    c = utility.inv_marg_func(wa,par)
    d = (par.chi*wb)/(wa-wb)-1
    
    # ii. states and value
    a = par.grid_a_pd_nd
    b = par.grid_b_pd_nd
    m,n,v = inv_mn_and_v(c,d,a,b,w,par)

    # iii. upperenvelope and interp to common
    upperenvelope.compute(out_c,out_d,out_v,m,n,c,d,v,num,w,par)

@njit
def solve_dcon(out_c,out_d,out_v,w,wa,par):

    num = 2

    # i. decisions                
    c = utility.inv_marg_func(wa,par)
    d = par.d_dcon
        
    # ii. states and value
    a = par.grid_a_pd_nd
    b = par.grid_b_pd_nd
    m,n,v = inv_mn_and_v(c,d,a,b,w,par)
                        
    # iii. value of deviating a bit from the constraint
    valt = np.zeros(v.shape)
    deviate_d_con(valt,n,c,a,w,par)
        
    # v. upperenvelope and interp to common
    upperenvelope.compute(out_c,out_d,out_v,m,n,c,d,v,num,w,par,valt)

@njit
def solve_acon(out_c,out_d,out_v,w,wb,par):

    num = 3

    # i. allocate
    c = np.zeros((par.Nc_acon,par.Nb_acon))
    d = np.zeros((par.Nc_acon,par.Nb_acon))
    a = np.zeros((par.Nc_acon,par.Nb_acon))
    b = np.zeros((par.Nc_acon,par.Nb_acon))
    w_acon = np.zeros((par.Nc_acon,par.Nb_acon))

    for i_b in range(par.Nb_acon):
        
        # ii. setup
        wb_acon = linear_interp.interp_2d(par.grid_b_pd,par.grid_a_pd,wb,par.b_acon[i_b],0)

        c_min = utility.inv_marg_func((par.chi+1)*wb_acon,par)
        c_max = utility.inv_marg_func(wb_acon,par)

        c[:,i_b] = nonlinspace_jit(c_min,c_max,par.Nc_acon,par.phi_m)
        
        # iii. choices
        d[:,i_b] = par.chi/(utility.marg_func(c[:,i_b],par)/(wb_acon)-1)-1
                        
        # iv. post-decision states and value function
        b[:,i_b] = par.b_acon[i_b]
        w_acon[:,i_b] = linear_interp.interp_2d(par.grid_b_pd,par.grid_a_pd,w,par.b_acon[i_b],0)
                
    # v. states and value
    m,n,v = inv_mn_and_v(c,d,a,b,w_acon,par)
                    
    # vi. upperenvelope and interp to common
    upperenvelope.compute(out_c,out_d,out_v,m,n,c,d,v,num,w,par)

@njit
def solve_con(out_c,out_d,out_v,w,par):
                        
    # i. choices
    c = par.grid_m_nd
    d = np.zeros(c.shape)
        
    # ii. post-decision states
    a = np.zeros(c.shape)
    b = par.grid_n_nd

    # iii. post decision value
    w_con = np.zeros(c.shape)
    linear_interp.interp_2d_vec(par.grid_b_pd,par.grid_a_pd,w,b.ravel(),a.ravel(),w_con.ravel())

    # iv. value
    v = utility.func(c,par) + w_con     

    out_c[:] = c
    out_d[:] = d
    out_v[:] = v

@njit
def solve(t,sol,par):

    w = sol.w[t]
    wa = sol.wa[t]
    wb = sol.wb[t]

    # a. solve each segment
    solve_ucon(sol.ucon_c[t,:,:],sol.ucon_d[t,:,:],sol.ucon_v[t,:,:],w,wa,wb,par)
    solve_dcon(sol.dcon_c[t,:,:],sol.dcon_d[t,:,:],sol.dcon_v[t,:,:],w,wa,par)
    solve_acon(sol.acon_c[t,:,:],sol.acon_d[t,:,:],sol.acon_v[t,:,:],w,wb,par)
    solve_con(sol.con_c[t,:,:],sol.con_d[t,:,:],sol.con_v[t,:,:],w,par)

    # b. upper envelope    
    seg_max = np.zeros(4)
    for i_n in range(par.Nn):
        for i_m in range(par.Nm):

            # i. find max
            seg_max[0] = sol.ucon_v[t,i_n,i_m]
            seg_max[1] = sol.dcon_v[t,i_n,i_m]
            seg_max[2] = sol.acon_v[t,i_n,i_m]
            seg_max[3] = sol.con_v[t,i_n,i_m]

            i = np.argmax(seg_max)

            # ii. over-arching optimal choices
            sol.inv_v[t,i_n,i_m] = -1.0/seg_max[i]

            if i == 0:
                sol.c[t,i_n,i_m] = sol.ucon_c[t,i_n,i_m]
                sol.d[t,i_n,i_m] = sol.ucon_d[t,i_n,i_m]
            elif i == 1:
                sol.c[t,i_n,i_m] = sol.dcon_c[t,i_n,i_m]
                sol.d[t,i_n,i_m] = sol.dcon_d[t,i_n,i_m]
            elif i == 2:
                sol.c[t,i_n,i_m] = sol.acon_c[t,i_n,i_m]
                sol.d[t,i_n,i_m] = sol.acon_d[t,i_n,i_m]
            elif i == 3:
                sol.c[t,i_n,i_m] = sol.con_c[t,i_n,i_m]
                sol.d[t,i_n,i_m] = sol.con_d[t,i_n,i_m]
        
    # c. derivatives 
    
    # i. m
    vm = utility.marg_func(sol.c[t],par)
    sol.inv_vm[t,:,:] = 1.0/vm

    # ii. n         
    a = par.grid_m_nd - sol.c[t] - sol.d[t]
    b = par.grid_n_nd + sol.d[t] + pens.func(sol.d[t],par)

    wb_now = np.zeros(a.shape)
    linear_interp.interp_2d_vec(par.grid_b_pd,par.grid_a_pd,wb,b.ravel(),a.ravel(),wb_now.ravel())
    
    vn = wb_now
    sol.inv_vn[t,:,:] = 1.0/vn