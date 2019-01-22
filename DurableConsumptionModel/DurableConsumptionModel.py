# -*- coding: utf-8 -*-
"""DurableConsumptionModel

Solves a consumption-saving model with a durable consumption good and non-convex adjustment costs with either:

A. vfi: value function iteration
B. nvfi: nested value function iteration
C. negm: nested endogenous grid point method

"""

##############
# 1. imports #
##############

import time
import numpy as np
from numba import boolean, int32, double

# consav package
from consav import linear_interp # for linear interpolation
from consav import golden_section_search # for optimization in 1D
from consav import misc # various consav
from consav.ConsumptionSavingModel import ConsumptionSavingModel # baseline model classes
from consav import upperenvelope # used in negm

# local modules
import utility
import trans
import last_period
import post_decision
import vfi
import nvfi
import negm
import simulate
import figs

class DurableConsumptionModelClass(ConsumptionSavingModel):
    
    #########
    # setup #
    #########

    def __init__(self,name='baseline',load=False,solmethod='vfi',**kwargs): # called when created
        """ basic setup

        Args:

        name (str,optional): name, used when saving/loading
        solmethod (str,optional): solmethod, used when solving
        load (bool,optinal): load from disc
            **kwargs: change to baseline parameter in .par

        Define parlist, sollist and simlist contain information on the
        model parameters and the variables when solving and simulating.

        Call .setup(**kwargs).

        """       

        self.name = name 
        self.solmethod = solmethod

        # a. define subclasses
        parlist = [
            ('T',int32),
            ('beta',double),
            ('rho',double),
            ('alpha',double),
            ('db_ubar',double),
            ('R',double),
            ('tau',double),
            ('delta',double),
            ('sigma_psi',double),
            ('Npsi',int32),
            ('sigma_xi',double),
            ('Nxi',int32),
            ('pi',double),
            ('mu',double),
            ('Nm',int32),
            ('grid_m',double[:]),
            ('m_max',double),
            ('Np',int32),
            ('grid_p',double[:]),    
            ('p_min',double),    
            ('p_max',double),    
            ('Ndb',int32),
            ('grid_db',double[:]),    
            ('db_max',double),
            ('Nx',int32),
            ('grid_x',double[:]),    
            ('x_max',double),    
            ('Na',int32),
            ('grid_a',double[:]),        
            ('a_max',double),        
            ('Nshocks',int32),        
            ('psi',double[:]),        
            ('psi_w',double[:]),        
            ('xi',double[:]),        
            ('xi_w',double[:]),        
            ('tol',double),
            ('do_print',boolean),
            ('do_print_t',boolean), 
            ('simT',int32),
            ('simN',int32),
            ('Nc_keep',int32),
            ('Nd_adj',int32),
            ('Nc_adj',int32),
            ('grid_c_keep',double[:]),
            ('grid_d_adj',double[:]),
            ('grid_c_adj',double[:]),
            ('t',int32),
            ('cppthreads',int32),
            ('sim_seed',int32),
        ]

        sollist = [
            ('c_keep',double[:,:,:,:]),
            ('inv_v_keep',double[:,:,:,:]),
            ('inv_v_adj',double[:,:,:]),
            ('c_adj',double[:,:,:]),
            ('d_adj',double[:,:,:]),
            ('inv_w',double[:,:,:,:]),
            ('q',double[:,:,:,:])
        ]

        simlist = [
            ('p',double[:,:]),
            ('db',double[:,:]),
            ('m',double[:,:]),
            ('x',double[:,:]),
            ('c',double[:,:]),
            ('d',double[:,:]),
            ('a',double[:,:]),
            ('psi',double[:,:]),
            ('xi',double[:,:]),
            ('discrete',double[:,:])
        ]

        # b. create subclasses
        self.par,self.sol,self.sim = self.create_subclasses(parlist,sollist,simlist)

        # c. load
        if load:
            self.load()
        else:
            self.setup(**kwargs)

    def setup(self,**kwargs):
        """ define baseline values and update

        Args:

             **kwargs: change to baseline parameter in .par

        """   

        # a. baseline parameters
        
        # horizon
        self.par.T = 5
        
        # preferences
        self.par.beta = 0.96
        self.par.rho = 2
        self.par.alpha = 0.9
        self.par.db_ubar = 1e-2

        # returns and income
        self.par.R = 1.03
        self.par.tau = 0.10
        self.par.delta = 0.15
        self.par.sigma_psi = 0.1
        self.par.Npsi = 2
        self.par.sigma_xi = 0.1
        self.par.Nxi = 2
        self.par.pi = 0.1
        self.par.mu = 0.5
        
        # grids
        self.par.Np = 200
        self.par.p_min = 1e-4
        self.par.p_max = 3
        self.par.Ndb = 200
        self.par.db_max = 3
        self.par.Nm = 200
        self.par.m_max = 10        
        self.par.Nx = 500
        self.par.x_max = 15
        self.par.Na = 500
        self.par.a_max = self.par.m_max
        self.par.Nc_keep = 500
        self.par.Nd_adj = 500
        self.par.Nc_adj = 500

        # misc
        self.par.tol = 1e-8
        self.par.do_print = False
        self.par.do_print_t = True
        self.par.cppthreads = 72

        # simulation
        self.par.simT = self.par.T
        self.par.simN = 100000
        self.par.sim_seed = 1998

        # b. update baseline parameters using keywords 
        for key,val in kwargs.items():
            setattr(self.par,key,val) # like par.key = val
        
        # c. setup_grids
        self.setup_grids()
        
    def setup_grids(self):
        """ construct grids for states and shocks """
        
        # a. states        
        self.par.grid_p = misc.nonlinspace(self.par.p_min,self.par.p_max,self.par.Np,1.1)
        self.par.grid_db = misc.nonlinspace(0,self.par.db_max,self.par.Ndb,1.1)
        self.par.grid_m = misc.nonlinspace(0,self.par.m_max,self.par.Nm,1.1)
        self.par.grid_x = misc.nonlinspace(0,self.par.x_max,self.par.Nx,1.1)
        
        # b. post-decision states
        self.par.grid_a = misc.nonlinspace(0,self.par.a_max,self.par.Na,1.1)
        
        # c. shocks
        shocks = misc.create_shocks(
            self.par.sigma_psi,self.par.Npsi,self.par.sigma_xi,self.par.Nxi,
            self.par.pi,self.par.mu)
        self.par.psi,self.par.psi_w,self.par.xi,self.par.xi_w,self.par.Nshocks = shocks

        # d. decision grids for vfi
        self.par.grid_c_keep = np.linspace(0,1,self.par.Nc_keep)
        self.par.grid_d_adj = np.linspace(0,1,self.par.Nd_adj)
        self.par.grid_c_adj = np.linspace(0,1,self.par.Nc_adj)

        # d. set seed
        np.random.seed(self.par.sim_seed)

    def checksum(self):
        """ calculate and print checksum """

        print('')
        print(f'checksum, inv_w: {np.mean(self.sol.inv_w[0]):.8f}')
        print(f'checksum, c_keep: {np.mean(self.sol.c_keep[0]):.8f}')
        print(f'checksum, d_adj: {np.mean(self.sol.d_adj[0]):.8f}')
        print(f'checksum, c_adj: {np.mean(self.sol.c_adj[0]):.8f}')
        print(f'checksum, inv_v_keep: {np.mean(self.sol.inv_v_keep[0]):.8f}')
        print(f'checksum, inv_v_adj: {np.mean(self.sol.inv_v_adj[0]):.8f}')
        print('')

    #########
    # solve #
    #########

    def numba_analysis(self):
        """ solve the model with very coarse grids and simulate with very few persons"""

        tic = time.time()

        # a. define
        fastpar = dict()
        fastpar['do_print'] = False
        fastpar['do_print_t'] = False
        fastpar['T'] = 2
        fastpar['Np'] = 3
        fastpar['Ndb'] = 3
        fastpar['Nm'] = 3
        fastpar['Nx'] = 3
        fastpar['Na'] = 3
        fastpar['Nc_keep'] = 3
        fastpar['Nd_adj'] = 3
        fastpar['Nc_adj'] = 3
        fastpar['simN'] = 2

        # b. apply
        for key,val in fastpar.items():
            prev = getattr(self.par,key)
            setattr(self.par,key,val)
            fastpar[key] = prev

        self.setup_grids()

        # c. solve
        self.solve(do_assert=False)

        # d. simulate
        self.solve(do_assert=False)

        # e. reiterate
        for key,val in fastpar.items():
            setattr(self.par,key,val)
        self.setup_grids()

        toc = time.time()
        print(f'numba-analysis done in {toc-tic:.1f} secs')

    def solve(self,do_assert=True,compiler='vs'):
        """ solve the model
        
        Args:

            do_assert (bool,optional): make assertions on the solution
            compiler (str,optional): compiler when solving with a _cpp method
        
        """

        # a. allocate solution
        keep_shape = (self.par.T,self.par.Np,self.par.Ndb,self.par.Nm)
        self.sol.c_keep = np.zeros(keep_shape)
        self.sol.inv_v_keep = np.zeros(keep_shape)

        adj_shape = (self.par.T,self.par.Np,self.par.Nx)
        self.sol.d_adj = np.zeros(adj_shape)
        self.sol.c_adj = np.zeros(adj_shape)
        self.sol.inv_v_adj = np.zeros(adj_shape)
        
        post_shape = (self.par.T-1,self.par.Np,self.par.Ndb,self.par.Na)
        self.sol.inv_w = np.nan*np.zeros(post_shape)
        self.sol.q = np.nan*np.zeros((self.par.T-1,self.par.Np,self.par.Ndb,self.par.Na))
        
        # b. compile if using cpp
        if self.solmethod in ['vfi_cpp']:
            self.setup_cpp(compiler=compiler)
            self.link_cpp('vfi',['solve_keep','solve_adj'])

        # c. backwards induction
        for t in reversed(range(self.par.T)):
            
            self.par.t = t
            tic = time.time()
            
            # i. last period
            if t == self.par.T-1:

                last_period.solve(t,self.sol,self.par)

                if do_assert:
                    assert np.all((self.sol.c_keep[t] >= 0) & (np.isnan(self.sol.c_keep[t]) == False))
                    assert np.all((self.sol.inv_v_keep[t] >= 0) & (np.isnan(self.sol.inv_v_keep[t]) == False))
                    assert np.all((self.sol.d_adj[t] >= 0) & (np.isnan(self.sol.d_adj[t]) == False))
                    assert np.all((self.sol.c_adj[t] >= 0) & (np.isnan(self.sol.c_adj[t]) == False))
                    assert np.all((self.sol.inv_v_adj[t] >= 0) & (np.isnan(self.sol.inv_v_adj[t]) == False))

            # ii. all other periods
            else:
                
                # o. compute post-decision functions
                tic_w = time.time()

                if self.solmethod in ['nvfi']:
                    post_decision.compute_wq(t,self.sol,self.par)
                elif self.solmethod in ['negm']:
                    post_decision.compute_wq(t,self.sol,self.par,compute_q=True)

                toc_w = time.time()
                if self.par.do_print:
                    print(f'  w computed in {toc_w-tic_w:.1f} secs')

                if do_assert and self.solmethod in ['nvfi','negm']:
                    assert np.all((self.sol.inv_w[t] > 0) & (np.isnan(self.sol.inv_w[t]) == False))
                    if self.solmethod in ['negm']:                                                       
                        assert np.all((self.sol.q[t] > 0) & (np.isnan(self.sol.q[t]) == False))

                # oo. solve keeper problem
                tic_keep = time.time()

                if self.solmethod == 'nvfi':                
                    nvfi.solve_keep(t,self.sol,self.par)
                elif self.solmethod == 'negm':
                    negm.solve_keep(t,self.sol,self.par)
                elif self.solmethod == 'vfi':
                    vfi.solve_keep(t,self.sol,self.par)
                elif self.solmethod == 'vfi_cpp':
                    self.call_cpp('vfi','solve_keep')

                toc_keep = time.time()
                if self.par.do_print:
                    print(f'  solved keeper problem in {toc_keep-tic_keep:.1f} secs')

                if do_assert:
                    assert np.all((self.sol.c_keep[t] >= 0) & (np.isnan(self.sol.c_keep[t]) == False))
                    assert np.all((self.sol.inv_v_keep[t] >= 0) & (np.isnan(self.sol.inv_v_keep[t]) == False))

                # ooo. solve adjuster problem
                tic_adj = time.time()

                if self.solmethod in ['nvfi','negm']:
                    nvfi.solve_adj(t,self.sol,self.par)                  
                elif self.solmethod == 'vfi':
                    vfi.solve_adj(t,self.sol,self.par)                  
                elif self.solmethod == 'vfi_cpp':
                    self.call_cpp('vfi','solve_adj')  

                toc_adj = time.time()
                if self.par.do_print:
                    print(f'  solved adjuster problem in {toc_adj-tic_adj:.1f} secs')

                if do_assert:
                    assert np.all((self.sol.d_adj[t] >= 0) & (np.isnan(self.sol.d_adj[t]) == False))
                    assert np.all((self.sol.c_adj[t] >= 0) & (np.isnan(self.sol.c_adj[t]) == False))
                    assert np.all((self.sol.inv_v_adj[t] >= 0) & (np.isnan(self.sol.inv_v_adj[t]) == False))

            # iii. print
            toc = time.time()
            if self.par.do_print or self.par.do_print_t:
                print(f' t = {t} solved in {toc-tic:.1f} secs')
        
        # d. compile if using cpp
        if self.solmethod in ['vfi_cpp']:
            self.delink_cpp('vfi')

    ############
    # simulate #
    ############

    def simulate(self):
        """ simulate the model """

        tic = time.time()

        # a. allocate
        sim_shape = (self.par.T+1,self.par.simN)
        self.sim.p = np.zeros(sim_shape)
        self.sim.m = np.zeros(sim_shape)
        self.sim.db = np.zeros(sim_shape)
        self.sim.x = np.zeros(sim_shape)
        self.sim.c = np.zeros(sim_shape)
        self.sim.d = np.zeros(sim_shape)
        self.sim.a = np.zeros(sim_shape)
        self.sim.discrete = np.zeros(sim_shape)

        # b. random shocks
        I = np.random.choice(self.par.psi.size,
            size=(self.par.T,self.par.simN), 
            p=self.par.psi_w*self.par.xi_w)
        self.sim.psi = self.par.psi[I]
        self.sim.xi = self.par.xi[I]

        # c. call
        self.par.simT = self.par.T
        simulate.lifecycle(self.sim,self.sol,self.par)

        toc = time.time()
        
        print(f'model simulated in {toc-tic:.1f} secs')

    ########
    # figs #
    ########

    def decision_functions(self):
        figs.decision_functions(self)

    def lifecycle(self):        
        figs.lifecycle(self)

if __name__ == "__main__":
    pass