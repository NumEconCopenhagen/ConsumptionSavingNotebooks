# -*- coding: utf-8 -*-
"""GEModelClass

Solves and simulates a buffer-stock consumption-saving problem for use in a general equilibrium model

"""

##############
# 1. imports #
##############

import time
import numpy as np
from numba import njit, prange

# consav
from consav import ModelClass, jit # baseline model class and jit
from consav import linear_interp # linear interpolation
from consav.grids import equilogspace # grids
from consav.markov import log_rouwenhorst # markov processes
from consav.misc import elapsed

############
# 2. model #
############

class GEModelClass(ModelClass):
    
    #########
    # setup #
    #########     
     
    def settings(self):
        """ fundamental settings """

        # for safe type inference
        self.not_floats = ['Ne','Na','max_iter_solve','max_iter_simulate','path_T']

    def setup(self):
        """ set baseline parameters """

        par = self.par
        
        # a. steady state values
        par.r_ss = np.nan
        par.w_ss = np.nan
        par.K_ss = np.nan
        par.Y_ss = np.nan
        par.C_ss = np.nan
        par.kd_ss = np.nan
        par.ks_ss = np.nan

        # b. preferences
        par.sigma = 1.0 # CRRA coefficient
        par.beta = 0.982 # discount factor

        # c. production
        par.Z = 1.0 # technology level in steady state
        par.Z_sigma = 0.01 # shock
        par.Z_rho = 0.90 # persistence

        par.alpha = 0.11 # Cobb-Douglas coefficient
        par.delta = 0.025 # depreciation rate

        # d. income parameters
        par.rho = 0.966 # AR(1) parameter
        par.sigma_e = 0.10 # std. of persistent shock
        par.Ne = 7 # number of states

        # e. grids         
        par.a_max = 200.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # f. misc.
        par.path_T = 500 # length of path
        par.max_iter_solve = 5000 # maximum number of iterations when solving
        par.max_iter_simulate = 5000 # maximum number of iterations when simulating
        par.solve_tol = 1e-10 # tolerance when solving
        par.simulate_tol = 1e-10 # tolerance when simulating

    def allocate(self):
        """ allocate model, i.e. create grids and allocate solution and simluation arrays """

        par = self.par
        sol = self.sol
        sim = self.sim

        # a. grids
        par.a_grid = np.zeros(par.Na)
        par.e_grid = np.zeros(par.Ne)
        par.e_trans = np.zeros((par.Ne,par.Ne))
        par.e_ergodic = np.zeros(par.Ne)
        par.e_trans_cumsum = np.zeros((par.Ne,par.Ne))
        par.e_ergodic_cumsum = np.zeros(par.Ne)

        self.create_grids()

        # b. solution
        sol_shape = (par.Ne,par.Na)
        sol.a = np.zeros(sol_shape)
        sol.m = np.zeros(sol_shape)
        sol.c = np.zeros(sol_shape)
        sol.Va = np.zeros(sol_shape)

        # path
        path_sol_shape = (par.path_T,par.Ne,par.Na)
        sol.path_a = np.zeros(path_sol_shape)
        sol.path_m = np.zeros(path_sol_shape)
        sol.path_c = np.zeros(path_sol_shape)
        sol.path_Va = np.zeros(path_sol_shape)

        # c. simulation
        sim_shape = sol_shape
        sim.i = np.zeros(sim_shape,dtype=np.int_)
        sim.w = np.zeros(sim_shape)
        sim.D = np.zeros(sim_shape)

        # path
        path_sim_shape = path_sol_shape
        sim.path_i = np.zeros(path_sim_shape,dtype=np.int_)
        sim.path_w = np.zeros(path_sim_shape)
        sim.path_D = np.zeros(path_sim_shape)
        sim.path_K = np.zeros(par.path_T)
        sim.path_C = np.zeros(par.path_T)

        # jacobians
        jac_shape = (par.path_T,par.path_T)

        sol.jac_curlyK_r = np.zeros(jac_shape)
        sol.jac_curlyK_w = np.zeros(jac_shape)
        sol.jac_C_r = np.zeros(jac_shape)
        sol.jac_C_w = np.zeros(jac_shape)

        sol.jac_r_K = np.zeros(jac_shape)
        sol.jac_w_K = np.zeros(jac_shape)
        sol.jac_r_Z = np.zeros(jac_shape)
        sol.jac_w_Z = np.zeros(jac_shape)
        
        sol.H_K = np.zeros(jac_shape)
        sol.H_Z = np.zeros(jac_shape)
        sol.G = np.zeros(jac_shape)
        
    def create_grids(self):
        """ construct grids for states and shocks """

        par = self.par

        # a. assets
        par.a_grid[:] = equilogspace(0,par.a_max,par.Na)

        # b. productivity
        e_objs = log_rouwenhorst(par.rho,par.sigma_e,par.Ne)
        par.e_grid[:] = e_objs[0]
        par.e_trans[:,:] = e_objs[1]
        par.e_ergodic[:] = e_objs[2]
        par.e_trans_cumsum[:,:] = e_objs[3]
        par.e_ergodic_cumsum[:] = e_objs[4]

    #########
    # solve #
    #########
    
    def get_path_Z(self):
        """ calculate Z path """

        par = self.par

        path_Z = np.ones(par.path_T)
        path_Z[0] = par.Z*(1+par.Z_sigma)
        for t in range(1,par.path_T):
            path_Z[t] = (1-par.Z_rho)*par.Z + par.Z_rho*path_Z[t-1]

        return path_Z

    def implied_r(self,k,Z):
        """ implied r given k = K/L and optimal firm behavior """

        par = self.par
        r = Z*par.alpha*k**(par.alpha-1)-par.delta
        return r

    def implied_w(self,r,Z):
        """ implied w given r and optimal firm behavior """

        par = self.par
        w = Z*(1.0-par.alpha)*((r+par.delta)/(Z*par.alpha))**(par.alpha/(par.alpha-1))
        return w

    def firm_demand(self,r,Z):
        """ firm demand for k = K/L given r and optimal firm behavior """

        par = self.par
        k = ((r+par.delta)/(Z*par.alpha))**(1/(par.alpha-1))
        return k

    def firm_production(self,k,Z):
        """ firm production """

        par = self.par
        return Z*k**par.alpha

    def steady_state(self,do_print=True):
        """ computate steady state statistics """

        par = self.par
        sol = self.sol
        sim = self.sim

        # a. firm
        par.w_ss = self.implied_w(par.r_ss,par.Z)
        par.kd_ss = self.firm_demand(par.r_ss,par.Z)
        par.Y_ss = self.firm_production(par.kd_ss,par.Z)

        # b. solve household problem
        self.solve_household_ss(par.r_ss,do_print=do_print)
        self.simulate_household_ss(do_print=do_print)

        # implied supply of capital and consumption
        par.ks_ss = np.sum(sim.D*sol.a)
        par.C_ss = np.sum(sim.D*sol.c)

        # c. equilibrium conditions
        par.K_ss = par.kd_ss
        if do_print:
            print('')
            print(f'r: {par.r_ss:.4f}')
            print(f'w: {par.w_ss:.4f}')
            print(f'Y: {par.Y_ss:.4f}')
            print(f'K/Y: {par.K_ss/par.Y_ss:.4f}')
            print('')
            print(f'capital market clearing: {par.ks_ss-par.kd_ss:12.8f}')
            print(f'goods market clearing: {par.Y_ss-par.C_ss-par.delta*par.K_ss:12.8f}')

    def solve_household_ss(self,r,Va=None,do_print=False):
        """ gateway for solving the model in steady state """
        
        t0 = time.time()

        with jit(self) as model:
            
            success = True
            try:

                par = model.par
                sol = model.sol
                
                # a. find wage from optimal firm behavior
                w = self.implied_w(r,par.Z)

                # b. create (or re-create) grids
                self.create_grids()

                # c. solve
                sol.m[:,:] = (1+r)*par.a_grid[np.newaxis,:] + w*par.e_grid[:,np.newaxis]
                sol.Va[:,:] = (1+r)*(0.1*sol.m)**(-par.sigma) if Va is None else Va

                it = 0
                while True:

                    # i. save
                    a_old = sol.a.copy()

                    # ii. egm
                    time_iteration(par,r,w,sol.Va,sol.Va,sol.a,sol.c,sol.m)

                    # ii. check
                    if np.max(np.abs(sol.a-a_old)) < par.solve_tol: break
                    
                    # iv. increment
                    it += 1
                    if it > par.max_iter_solve: raise Exception('too many iterations when solving for steady state')

            except:

                success = False
                
        if do_print:
            print(f'household problem solved in {elapsed(t0)} [{it} iterations]')

        return success

    def solve_household_path(self,path_r,path_w,do_print=False):
        """ gateway for solving the model along price path (with optimal update of path for w) """

        t0 = time.time()

        with jit(self) as model:

            par = model.par
            sol = model.sol

            # a. create (or re-create) grids
            self.create_grids()

            # c. solve
            for t in reversed(range(par.path_T)):
                
                # i. prices
                r = path_r[t]
                w = path_w[t]

                # ii. next-period
                if t == par.path_T-1:
                    Va_p = sol.Va
                else:
                    Va_p = sol.path_Va[t+1]

                # ii. solve
                sol.path_m[t] = (1+r)*par.a_grid[np.newaxis,:] + w*par.e_grid[:,np.newaxis]

                # iii. time iteration
                time_iteration(par,r,w,Va_p,sol.path_Va[t],sol.path_a[t],sol.path_c[t],sol.path_m[t])

        if do_print:
            print(f'household problem solved in {elapsed(t0)}')

    def simulate_household_ss(self,D=None,do_print=False):
        """ gateway for simulating the model towards the steady state"""
        
        t0 = time.time()

        with jit(self) as model:

            success = True
            try:

                par = model.par
                sol = model.sol
                sim = model.sim        

                # a. intial guess
                D = (np.repeat(par.e_ergodic,par.Na)/par.Na).reshape(par.Ne,par.Na) if D is None else D

                # b. simulate
                it = simulate_ss(par,sol,sim,D)

                if do_print:
                    print(f'household problem simulated in {elapsed(t0)} [{it} iterations]')

            except:

                success = False

        return success
        
    def simulate_household_path(self,D0,do_print=False):
        """ gateway for simulating the model along path"""
        
        t0 = time.time()

        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim        

            simulate_path(par,sol,sim,D0)

        if do_print:
            print(f'household problem simulated in {elapsed(t0)}')

######################
# fast jit functions #
######################

@njit(parallel=True)        
def time_iteration(par,r,w,Va_p,Va,a,c,m):
    """ perform time iteration step with Va_p from previous iteration """

    # a. post-decision 
    marg_u_plus = (par.beta*par.e_trans)@Va_p

    # b. egm loop
    for i_e in prange(par.Ne):
        
        # i. egm
        c_endo = marg_u_plus[i_e]**(-1/par.sigma)
        m_endo = c_endo + par.a_grid

        # ii. interpolation
        linear_interp.interp_1d_vec(m_endo,par.a_grid,m[i_e],a[i_e])
        a[i_e,0] = np.fmax(a[i_e,0],0)
        c[i_e] = m[i_e]-a[i_e]

        # iii. envelope condition
        Va[i_e] = (1+r)*c[i_e]**(-par.sigma)
    
@njit
def binary_search(imin,x,xi):
    """ binary search algorithm """

    Nx = x.size

    # a. checks
    if xi <= x[0]:
        return 0
    elif xi >= x[Nx-2]:
        return Nx-2
    
    # b. binary search
    half = Nx//2
    while half:
        imid = imin + half
        if x[imid] <= xi:
            imin = imid
        Nx -= half
        half = Nx//2
        
    return imin

@njit(parallel=True) 
def find_i_and_w(par,sim,a):
    """ find indices and weights for simulation """

    i = sim.i
    w = sim.w

    for i_e in prange(par.Ne):
        for i_a in prange(par.Na):
            
            # a. policy
            a_ = a[i_e,i_a]

            # b. find i_ such par.a_grid[i_] <= a_ < par.a_grid[i_+1]
            i_ = i[i_e,i_a] = binary_search(0,par.a_grid,a_) 

            # b. weight
            w[i_e,i_a] = (par.a_grid[i_+1] - a_) / (par.a_grid[i_+1] - par.a_grid[i_])

@njit(parallel=True)   
def simulate(par,sim,e_trans_T,D):
    """ simulate given weight indices are weight """

    # a. assuming e is constant
    D_upd = np.zeros(D.shape)
    for i_e in prange(par.Ne):
        for i_a in range(par.Na):
            
            # a. from
            D_ = D[i_e,i_a]
            w = sim.w[i_e,i_a]

            # b. to
            i = sim.i[i_e,i_a]            
            D_upd[i_e,i] += D_*w
            D_upd[i_e,i+1] += D_*(1.0-w)
    
    # b. account for transition of e
    D_upd_ = e_trans_T@D_upd

    return D_upd_

@njit(parallel=True)        
def simulate_ss(par,sol,sim,D):
    """ simulate forward to steady state """

    # a. indices and weights
    find_i_and_w(par,sim,sol.a)

    # b. iterate
    e_trans_T = par.e_trans.T.copy()
    it = 0
    while True:

        # i. update distribution
        D_upd = simulate(par,sim,e_trans_T,D)

        # ii. check
        if np.max(np.abs(D_upd-D)) < par.simulate_tol: break
        D = D_upd

        # iii. increment
        it += 1
        if it > par.max_iter_simulate: raise Exception()

    # d. save
    sim.D[:,:] = D
    return it
    
@njit    
def simulate_path(par,sol,sim,D0):
    """ simulate along path """

    e_trans_T = par.e_trans.T.copy()
    for t in range(par.path_T):

        # a. update distribution
        if t == 0:
            sim.path_D[t] = D0 # initial distribution is fixed
        else:
            find_i_and_w(par,sim,sol.path_a[t-1])
            sim.path_D[t] = simulate(par,sim,e_trans_T,sim.path_D[t-1])

        # b. capital
        sim.path_K[t] = np.sum(sol.path_a[t]*sim.path_D[t])
        sim.path_C[t] = np.sum(sol.path_c[t]*sim.path_D[t])