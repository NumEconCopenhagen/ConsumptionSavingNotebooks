# -*- coding: utf-8 -*-
"""ConsumptionSavingModel

Solves the Deaton-Carroll buffer-stock consumption model with vfi or egm:

"""

##############
# 1. imports #
##############

import time
import numpy as np
from numba import njit, prange
from scipy import optimize

# consav package
from consav import ModelClass, jit # baseline model class and jit
from consav import linear_interp # linear interpolation
from consav.grids import nonlinspace # grids
from consav.quadrature import log_normal_gauss_hermite # income shocks
from consav.misc import elapsed

import figs

############
# 2. model #
############

class ConsumptionSavingModelClass(ModelClass):
    
    #########
    # setup #
    #########      

    def settings(self):
        """ fundamental settings """

        # for safe type inference
        self.not_floats = ['solmethod','T','TR','age_min','Nxi','Npsi','Nm','Na',
                           'simT','simN','simlifecycle','Nshocks','do_cev']
    
    def setup(self):
        """ set baseline parameters """

        par = self.par
        par.solmethod = 'egm'
        
        # a. demographics
        par.T = 200
        par.TR = par.T # retirement age (end-of-period), no retirement if TR = T
        par.age_min = 25 # only relevant for figures

        # b. preferences
        par.rho = 2.0 # CRRA coeficient
        par.beta = 0.96 # discount factor

        # c. income parameters

        # growth
        par.G = 1.02 # growth factor

        # standard deviations
        par.sigma_xi = 0.1 # std. of transitory shock
        par.sigma_psi = 0.1 # std. of permanent shock

        # low income shock
        par.pi = 0.005 # probability of low income shock
        par.mu = 0.0 # value of low income shock

        # life-cycle
        par.L = np.ones(par.T) # if ones then no life-cycle           

        # d. saving and borrowing
        par.R = 1.04 # return factor
        par.borrowingfac = 0.0 # scales borrowing constraints

        # e. certainty equivalence (CEV): (1+cev) is multiplied with consumption in utility function
        par.cev = 0.0
        par.do_cev = 0 

        # f. numerical integration and grids         
        par.a_max = 20.0 # maximum point i grid for a
        par.a_phi = 1.1 # curvature parameters
        par.m_max = 20.0 # maximum point i grid for m
        par.m_phi = 1.1 # curvature parameters

        # number of elements
        par.Nxi  = 8 # number of quadrature points for xi
        par.Npsi = 8 # number of quadrature points for psi
        par.Na = 500 # number of points in grid for a
        par.Nm = 100 # number of points in grid for m

        # g. simulation
        par.sim_mini = 2.5 # initial m in simulation
        par.simN = 100_000 # number of persons in simulation
        par.simT = 100 # number of periods in simulation
        par.simlifecycle = 0 # = 0 simulate infinite horizon model

    def allocate(self):
        """ allocate model, i.e. create grids and allocate solution and simluation arrays """

        par = self.par
        sol = self.sol
        sim = self.sim

        # a. grids
        self.create_grids()

        # a. solution
        if par.solmethod == 'egm':

            sol_shape = (par.T,par.Na+1)
            sol.m = np.zeros(sol_shape)
            sol.c = np.zeros(sol_shape)
            sol.inv_v = np.zeros(sol_shape)

        elif par.solmethod == 'vfi':

            sol_shape = (par.T,par.Nm)
            sol.m = np.zeros(sol_shape)
            sol.c = np.zeros(sol_shape)
            sol.inv_v = np.zeros(sol_shape)

        # b. simulation
        sim_shape = (par.simN,par.simT)

        sim.m = np.zeros(sim_shape)
        sim.c = np.zeros(sim_shape)
        sim.a = np.zeros(sim_shape)
        sim.p = np.zeros(sim_shape)
        sim.y = np.zeros(sim_shape)
        sim.psi = np.zeros(sim_shape)
        sim.xi = np.zeros(sim_shape)
        sim.P = np.zeros(sim_shape)
        sim.Y = np.zeros(sim_shape)
        sim.M = np.zeros(sim_shape)
        sim.C = np.zeros(sim_shape)
        sim.A = np.zeros(sim_shape)

    def create_grids(self):
        """ construct grids for states and shocks """

        par = self.par

        # b. shocks

        # i. basic GuassHermite
        psi, psi_w = log_normal_gauss_hermite(sigma=par.sigma_psi,n=par.Npsi)
        xi, xi_w = log_normal_gauss_hermite(sigma=par.sigma_xi,n=par.Nxi)

        # ii. add low income shock to xi
        if par.pi > 0:
            
            # a. weights
            xi_w *= (1.0-par.pi)
            xi_w = np.insert(xi_w,0,par.pi)

            # b. values
            xi = (xi-par.mu*par.pi)/(1.0-par.pi)
            xi = np.insert(xi,0,par.mu)

        # iii. vectorize tensor product of shocks and total weight
        psi_vec,xi_vec = np.meshgrid(psi,xi,indexing='ij')
        psi_w_vec,xi_w_vec = np.meshgrid(psi_w,xi_w,indexing='ij')
        
        par.psi_vec = psi_vec.ravel()
        par.xi_vec = xi_vec.ravel()
        par.w = xi_w_vec.ravel()*psi_w_vec.ravel()

        assert 1-np.sum(par.w) < 1e-8 # == summing to 1

        # iv. count number of shock nodes
        par.Nshocks = par.w.size

        # c. minimum a
        if par.borrowingfac == 0:

            par.a_min = np.zeros(par.T) # never any borriwng

        else:

            # using formula from slides 
            psi_min = np.min(par.psi_vec)
            xi_min = np.min(par.xi_vec)
            
            par.a_min = np.ones(par.T)
            for t in reversed(range(par.T-1)):
                
                if t >= par.TR-1: # in retirement
                    Omega = 0
                elif t == par.TR-2: # next period is retirement
                    Omega = par.R**(-1)*par.G*par.L[t+1]*psi_min*xi_min
                else: # before retirement
                    Omega = par.R**(-1)*(np.fmin(Omega,par.borrowingfac)+xi_min)*par.G*par.L[t+1]*psi_min

                par.a_min[t] = -np.fmin(Omega,par.borrowingfac)*par.G*par.L[t+1]*psi_min
            
        # d. end-of-period assets and cash-on-hand
        par.grid_a = np.ones((par.T,par.Na))
        par.grid_m = np.ones((par.T,par.Nm))
        for t in range(par.T):
            par.grid_a[t,:] = nonlinspace(par.a_min[t]+1e-6,par.a_max,par.Na,par.a_phi)
            par.grid_m[t,:] = nonlinspace(par.a_min[t]+1e-6,par.m_max,par.Nm,par.m_phi)        

        # e. conditions
        par.FHW = np.float(par.G/par.R)
        par.AI = np.float((par.R*par.beta)**(1/par.rho))
        par.GI = np.float(par.AI*np.sum(par.w*par.psi_vec**(-1))/par.G)
        par.RI = np.float(par.AI/par.R)
        par.WRI = np.float(par.pi**(1/par.rho)*par.AI/par.R)
        par.FVA = np.float(par.beta*np.sum(par.w*(par.G*par.psi_vec)**(1-par.rho)))

        # g. check for existance of solution
        self.check(do_print=False)
    
    def check(self,do_print=True):
        """ check parameters """

        par = self.par

        if do_print:
            print(f'FHW = {par.FHW:.3f}, AI = {par.AI:.3f}, GI = {par.GI:.3f}, RI = {par.RI:.3f}, WRI = {par.WRI:.3f}, FVA = {par.FVA:.3f}')

        # check for existance of solution
        if par.sigma_xi == 0 and par.sigma_psi == 0 and par.pi == 0: # no risk
            if par.GI >= 1 and par.RI >= 1:
                raise Exception('GI >= 1 and RI >= 1: no solution')
        else:
            if par.FVA >= 1 or par.WRI >= 1:
                raise Exception('FVA >= 1 or WRI >= 1: no solution')
                                     
    #########
    # solve #
    #########
    
    def solve(self,do_print=True):
        """ gateway for solving the model """

        par = self.par

        # a. reset solution and simulation arrays
        self.allocate()

        # b. solve
        if par.solmethod == 'egm':
            self.solve_egm(do_print=do_print)
        elif par.solmethod == 'vfi':
            self.solve_vfi(do_print=do_print)
        else:
            raise Exception(f'{par.solmethod} is an unknown solution method')

    def solve_egm(self,do_print):
        """ solve the model using egm """
        
        t0 = time.time()

        with jit(self) as model:

            par = model.par
            sol = model.sol
            
            
            # a. allocate working memory
            m = np.zeros(par.Na)
            c = np.zeros(par.Na)
            inv_v = np.zeros(par.Na)

            # b. last period (= consume all)
            sol.m[-1,:] = np.linspace(0,par.a_max,par.Na+1)
            sol.c[-1,:] = sol.m[-1,:]
            sol.inv_v[-1,0] = 0
            sol.inv_v[-1,1:] = 1.0/utility(sol.c[-1,1:],par)

            # c. before last period
            for t in reversed(range(par.T-1)):
                
                # i. solve by EGM
                egm(par,sol,t,m,c,inv_v)

                # ii. add zero consumption
                sol.m[t,0] = par.a_min[t]
                sol.m[t,1:] = m
                sol.c[t,0] = 0
                sol.c[t,1:] = c
                sol.inv_v[t,0] = 0
                sol.inv_v[t,1:] = inv_v

        if do_print:
            print(f'model solved in {elapsed(t0)}')

    def solve_vfi(self,do_print):
        """ solve the model with vfi """

        t0 = time.time()
        
        with jit(self) as model:

            par = model.par
            sol = model.sol

            # a. last period (= consume all)
            sol.m[-1,:] = par.grid_m[-1,:]
            sol.c[-1,:] = sol.m[-1,:]
            for i,c in enumerate(sol.c[-1,:]):
                sol.inv_v[-1,i] = 1.0/utility(c,par)
            
            # b. before last period
            for t in reversed(range(par.T-1)):
                for i_m in range(par.Nm):

                    m = par.grid_m[t,i_m]

                    obj = lambda c: self.value_of_choice(c,t,m)
                    result = optimize.minimize_scalar(obj,method='bounded',bounds=(0,m))

                    sol.c[t,i_m] = result.x
                    sol.inv_v[t,i_m]= -1.0/result.fun
                
                # save grid for m
                sol.m[t,:] = par.grid_m[t,:]

        if do_print:
            print(f'model solved in {elapsed(t0)}')

    def value_of_choice(self,c,t,m):
        """ value of choice of c used in vfi """

        par = self.par
        sol = self.sol

        # a. end-of-period assets
        a = m-c

        # b. next-period cash-on-hand
        still_working_next_period = t+1 <= par.TR-1
        if still_working_next_period:
            fac = par.G*par.L[t]*par.psi_vec
            w = par.w
            xi = par.xi_vec
        else:
            fac = par.G*par.L[t]
            w = 1
            xi = 1

        m_plus = (par.R/fac)*a + xi            

        # c. continuation value
        if still_working_next_period:
            inv_v_plus = np.zeros(m_plus.size)
            linear_interp.interp_1d_vec(sol.m[t+1,:],sol.inv_v[t+1,:],m_plus,inv_v_plus)
        else:
            inv_v_plus = linear_interp.interp_1d(sol.m[t+1,:],sol.inv_v[t+1,:],m_plus)
        v_plus = 1/inv_v_plus
        
        # d. value-of-choice
        total = utility(c,par) + par.beta*np.sum(w*fac**(1-par.rho)*v_plus)
        return -total

    ############
    # simulate #
    ############
    
    def simulate(self,do_print=True,seed=2017):
        """ simulate the model """

        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim

            t0 = time.time()

            # a. set seed
            if not seed is None: np.random.seed(seed)

            # b. shocks
            _shocki = np.random.choice(par.Nshocks,size=(par.simN,par.simT),p=par.w)
            sim.psi[:] = par.psi_vec[_shocki]
            sim.xi[:] = par.xi_vec[_shocki]

            # c. initial values
            sim.m[:,0] = par.sim_mini 
            sim.p[:,0] = 0.0 

            # d. simulation
            simulate_time_loop(par,sol,sim)

            # e. renomarlized
            sim.P[:,:] = np.exp(sim.p)
            sim.Y[:,:] = np.exp(sim.y)
            sim.M[:,:] = sim.m*sim.P
            sim.C[:,:] = sim.c*sim.P
            sim.A[:,:] = sim.a*sim.P

            if do_print:
                print(f'model simulated in {elapsed(t0)}')
    
    #########
    # plots #
    #########
    
    plot_value_function_convergence = figs.plot_value_function_convergence
    plot_consumption_function_convergence = figs.plot_consumption_function_convergence
    plot_consumption_function_convergence_age = figs.plot_consumption_function_convergence_age
    plot_consumption_function_pf = figs.plot_consumption_function_pf
    plot_buffer_stock_target = figs.plot_buffer_stock_target
    plot_simulate_cdf_cash_on_hand = figs.plot_simulate_cdf_cash_on_hand
    plot_simulate_consumption_growth = figs.plot_simulate_consumption_growth
    plot_life_cycle_income = figs.plot_life_cycle_income
    plot_life_cycle_cashonhand = figs.plot_life_cycle_cashonhand
    plot_life_cycle_consumption = figs.plot_life_cycle_consumption
    plot_life_cycle_assets = figs.plot_life_cycle_assets

##############
# 3. utility #
##############

@njit
def utility(c,par):
    if par.do_cev:
        c_cev = c*(1.0 + par.cev)
        return c_cev**(1-par.rho)/(1-par.rho)

    else:
        return c**(1-par.rho)/(1-par.rho)    

@njit
def marg_utility(c,par):
    if par.do_cev:
        fac_cev = (1.0 + par.cev)**(1-par.rho)/(1-par.rho)
        return c**(-par.rho) * fac_cev    

    else:
        return c**(-par.rho)      

@njit
def inv_marg_utility(u,par):
    if par.do_cev:
        fac_cev = (1.0 + par.cev)**(1-par.rho)/(1-par.rho)
        return (u/fac_cev)**(-1/par.rho)   

    else:
        return u**(-1/par.rho)   

##########
# 4. egm #
##########

@njit(parallel=True)
def egm(par,sol,t,m,c,inv_v):
    """ apply egm step """

    # loop over end-of-period assets
    for i_a in prange(par.Na): # parallel

        a = par.grid_a[t,i_a]
        still_working_next_period = t+1 <= par.TR-1
        Nshocks = par.Nshocks if still_working_next_period else 1

        # loop over shocks
        avg_marg_u_plus = 0
        avg_v_plus = 0
        for i_shock in range(Nshocks):
            
            # a. prep
            if still_working_next_period:
                fac = par.G*par.L[t]*par.psi_vec[i_shock]
                w = par.w[i_shock]
                xi = par.xi_vec[i_shock]
            else:
                fac = par.G*par.L[t]
                w = 1
                xi = 1
        
            inv_fac = 1.0/fac

            # b. future m and c
            m_plus = inv_fac*par.R*a + xi
            c_plus = linear_interp.interp_1d(sol.m[t+1,:],sol.c[t+1,:],m_plus)
            inv_v_plus = linear_interp.interp_1d(sol.m[t+1,:],sol.inv_v[t+1,:],m_plus)
            v_plus = 1.0/inv_v_plus

            # c. average future marginal utility
            marg_u_plus = marg_utility(fac*c_plus,par)
            avg_marg_u_plus += w*marg_u_plus
            avg_v_plus += w*(fac**(1-par.rho))*v_plus

        # d. current c
        c[i_a] = inv_marg_utility(par.beta*par.R*avg_marg_u_plus,par)

        # e. current m
        m[i_a] = a + c[i_a]

        # f. current v
        if c[i_a] > 0:
            inv_v[i_a] = 1.0/(utility(c[i_a],par) + par.beta*avg_v_plus)
        else:
            inv_v[i_a] = 0

#################
# 5. simulation #
#################

@njit(parallel=True)
def simulate_time_loop(par,sol,sim):
    """ simulate model with parallization over households """

    # unpack (helps numba)
    m = sim.m
    p = sim.p
    y = sim.y
    c = sim.c
    a = sim.a

    sol_c = sol.c
    sol_m = sol.m

    # loop over first households and then time
    for i in prange(par.simN):
        for t in range(par.simT):
            
            # a. solution
            if par.simlifecycle == 0:
                grid_m = sol_m[0,:]
                grid_c = sol_c[0,:]
            else:
                grid_m = sol_m[t,:]
                grid_c = sol_c[t,:]
            
            # b. consumption
            c[i,t] = linear_interp.interp_1d(grid_m,grid_c,m[i,t])
            a[i,t] = m[i,t] - c[i,t]

            # c. next-period
            if t < par.simT-1:

                if t+1 > par.TR-1:
                    m[i,t+1] = par.R*a[i,t] / (par.G*par.L[t]) +  1
                    p[i,t+1] = np.log(par.G) + np.log(par.L[t]) + p[i,t]
                    y[i,t+1] = p[i,t+1]
                else:
                    m[i,t+1] = par.R*a[i,t] / (par.G*par.L[t]*sim.psi[i,t+1]) + sim.xi[i,t+1]
                    p[i,t+1] = np.log(par.G) + np.log(par.L[t]) + p[i,t] + np.log(sim.psi[i,t+1])   
                    if sim.xi[i,t+1] > 0:
                        y[i,t+1] = p[i,t+1] + np.log(sim.xi[i,t+1])
                    else:
                        y[i,t+1] = -np.inf