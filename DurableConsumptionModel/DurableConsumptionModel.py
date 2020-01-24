# -*- coding: utf-8 -*-
"""DurableConsumptionModel

Solves a consumption-saving model with a durable consumption good and non-convex adjustment costs with either:

A. vfi: value function iteration (only i c++)
B. nvfi: nested value function iteration (both in Python and c++)
C. negm: nested endogenous grid point method (both in Python and c++)

The do_2d switch turns on the extended version with two durable stocks.

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
from consav import misc # various functions
from consav import upperenvelope # used in negm
from consav import ModelClass # baseline model classes

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

class DurableConsumptionModelClass(ModelClass):
    
    #########
    # setup #
    #########

    def __init__(self,name='baseline',load=False,solmethod='negm',compiler='vs',**kwargs):
        """ basic setup

        Args:

        name (str,optional): name, used when saving/loading
        load (bool,optinal): load from disc
        solmethod (str,optional): solmethod, used when solving
        compiler (str,optional): compiler, 'vs' or 'intel' (used for c++)
        **kwargs: change to baseline parameter in .par

        Define parlist, sollist and simlist contain information on the
        model parameters and the variables when solving and simulating.

        Call .setup(**kwargs).

        """       

        self.name = name 
        self.solmethod = solmethod
        self.compiler = compiler
        self.vs_path = 'C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/'
        self.intel_path = 'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.5.274/windows/bin/'
        self.intel_vs_version = 'vs2017'
        
        # a. define subclasses
        parlist = [
            ('do_2d',boolean),
            ('T',int32),
            ('t',int32),
            ('beta',double),
            ('rho',double),
            ('alpha',double),
            ('d_ubar',double),
            ('d1_ubar',double),
            ('d2_ubar',double),
            ('R',double),
            ('tau',double),
            ('tau1',double),
            ('tau2',double),
            ('delta',double),
            ('delta1',double),
            ('delta2',double),
            ('gamma',double),
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
            ('Nn',int32),
            ('grid_n',double[:]),    
            ('n_max',double),
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
            ('sigma_p0',double),        
            ('mu_d0',double),        
            ('sigma_d0',double),        
            ('mu_a0',double),        
            ('sigma_a0',double),        
            ('simT',int32),
            ('simN',int32),
            ('sim_seed',int32),
            ('euler_cutoff',double),
            ('tol',double),
            ('do_print',boolean),
            ('do_print_period',boolean), 
            ('cppthreads',int32),
            ('do_marg_u',boolean),
            ('do_simple_wq',boolean),
            ('use_gs_in_vfi',boolean),            
            ('time_w',double[:]),
            ('time_keep',double[:]),
            ('time_adj',double[:]),
            ('time_adj_full',double[:]),
            ('time_adj_d1',double[:]),
            ('time_adj_d2',double[:]),
        ]

        sollist = [
            ('c_keep',double[:,:,:,:]),
            ('inv_v_keep',double[:,:,:,:]),
            ('inv_marg_u_keep',double[:,:,:,:]),
            ('inv_v_adj',double[:,:,:]),
            ('inv_marg_u_adj',double[:,:,:]),
            ('c_adj',double[:,:,:]),
            ('d_adj',double[:,:,:]),
            ('inv_w',double[:,:,:,:]),
            ('q',double[:,:,:,:]),
            ('q_c',double[:,:,:,:]),
            ('q_m',double[:,:,:,:]),
            ('c_keep_2d',double[:,:,:,:,:]),
            ('inv_v_keep_2d',double[:,:,:,:,:]),
            ('inv_marg_u_keep_2d',double[:,:,:,:,:]),
            ('inv_v_adj_full_2d',double[:,:,:]),
            ('inv_marg_u_adj_full_2d',double[:,:,:]),
            ('c_adj_full_2d',double[:,:,:]),
            ('d1_adj_full_2d',double[:,:,:]),
            ('d2_adj_full_2d',double[:,:,:]),
            ('inv_v_adj_d1_2d',double[:,:,:,:]),
            ('inv_marg_u_adj_d1_2d',double[:,:,:,:]),
            ('c_adj_d1_2d',double[:,:,:,:]),
            ('d1_adj_d1_2d',double[:,:,:,:]),
            ('inv_v_adj_d2_2d',double[:,:,:,:]),
            ('inv_marg_u_adj_d2_2d',double[:,:,:,:]),
            ('c_adj_d2_2d',double[:,:,:,:]),
            ('d2_adj_d2_2d',double[:,:,:,:]),
            ('inv_w_2d',double[:,:,:,:,:]),
            ('q_2d',double[:,:,:,:,:]),
            ('q_c_2d',double[:,:,:,:,:]),
            ('q_m_2d',double[:,:,:,:,:])

        ]

        simlist = [
            ('utility',double[:]),
            ('p0',double[:]),
            ('d0',double[:]),
            ('d10',double[:]),
            ('d20',double[:]),
            ('a0',double[:]),
            ('p',double[:,:]),
            ('n',double[:,:]), 
            ('n1',double[:,:]), 
            ('n2',double[:,:]), 
            ('m',double[:,:]),
            ('c',double[:,:]),
            ('d',double[:,:]),
            ('d1',double[:,:]),
            ('d2',double[:,:]),
            ('a',double[:,:]),
            ('psi',double[:,:]),
            ('xi',double[:,:]),
            ('discrete',int32[:,:]),
            ('euler_error',double[:,:]),
            ('euler_error_c',double[:,:]),
            ('euler_error_rel',double[:,:])
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
        
        self.par.do_2d = False

        # horizon
        self.par.T = 5
        
        # preferences
        self.par.beta = 0.965
        self.par.rho = 2
        self.par.alpha = 0.9
        self.par.d_ubar = 1e-2
        self.par.d1_ubar = 1e-2
        self.par.d2_ubar = 1e-2

        # returns and income
        self.par.R = 1.03
        self.par.tau = 0.10
        self.par.tau1 = 0.08
        self.par.tau2 = 0.12
        self.par.delta = 0.15
        self.par.delta1 = 0.10
        self.par.delta2 = 0.20
        self.par.gamma = 0.50 # note: the last_period.solve_2d function must be updated if this value is changed
        self.par.sigma_psi = 0.1
        self.par.Npsi = 5
        self.par.sigma_xi = 0.1
        self.par.Nxi = 5
        self.par.pi = 0.0
        self.par.mu = 0.5
        
        # grids
        self.par.Np = 50
        self.par.p_min = 1e-4
        self.par.p_max = 3
        self.par.Nn = 50
        self.par.n_max = 3
        self.par.Nm = 100
        self.par.m_max = 10        
        self.par.Nx = 100
        self.par.x_max = self.par.m_max + self.par.n_max
        self.par.Na = 100
        self.par.a_max = self.par.m_max+1

        # simulation
        self.par.sigma_p0 = 0.2
        self.par.mu_d0 = 0.8
        self.par.sigma_d0 = 0.2
        self.par.mu_a0 = 0.2
        self.par.sigma_a0 = 0.1
        self.par.simN = 100000
        self.par.sim_seed = 1998
        self.par.euler_cutoff = 0.02

        # misc
        self.par.tol = 1e-8
        self.par.do_print = False
        self.par.do_print_period = False
        self.par.cppthreads = 8
        self.par.do_simple_wq = False # not using optimized interpolation in C++
        self.par.do_marg_u = False # calculate marginal utility for use in egm

        # b. update baseline parameters using keywords 
        for key,val in kwargs.items():
            setattr(self.par,key,val) # like par.key = val
        self.par.simT = self.par.T

        # c. setup grids
        self.setup_grids()
        
    def setup_grids(self):
        """ construct grids for states and shocks """
        
        if self.solmethod in ['negm','negm_cpp','negm_2d_cpp']:
            self.par.do_marg_u = True

        # a. states        
        self.par.grid_p = misc.nonlinspace(self.par.p_min,self.par.p_max,self.par.Np,1.1)
        self.par.grid_n = misc.nonlinspace(0,self.par.n_max,self.par.Nn,1.1)
        self.par.grid_m = misc.nonlinspace(0,self.par.m_max,self.par.Nm,1.1)
        self.par.grid_x = misc.nonlinspace(0,self.par.x_max,self.par.Nx,1.1)
        
        # b. post-decision states
        self.par.grid_a = misc.nonlinspace(0,self.par.a_max,self.par.Na,1.1)
        
        # c. shocks
        shocks = misc.create_shocks(
            self.par.sigma_psi,self.par.Npsi,self.par.sigma_xi,self.par.Nxi,
            self.par.pi,self.par.mu)
        self.par.psi,self.par.psi_w,self.par.xi,self.par.xi_w,self.par.Nshocks = shocks

        # d. set seed
        np.random.seed(self.par.sim_seed)

        # e. timing
        self.par.time_w = np.zeros(self.par.T)
        self.par.time_keep = np.zeros(self.par.T)
        self.par.time_adj = np.zeros(self.par.T)
        self.par.time_adj_full = np.zeros(self.par.T)
        self.par.time_adj_d1 = np.zeros(self.par.T)
        self.par.time_adj_d2 = np.zeros(self.par.T)

    def checksum(self,simple=False,T=1):
        """ calculate and print checksum """

        if simple:
            if self.par.do_2d:
                print(f'checksum, inv_v_keep: {np.mean(self.sol.inv_v_keep_2d[0]):.8f}')
                print(f'checksum, inv_v_adj_full: {np.mean(self.sol.inv_v_adj_full_2d[0]):.8f}')
                print(f'checksum, inv_v_adj_d1_2d: {np.mean(self.sol.inv_v_adj_d1_2d[0]):.8f}')
                print(f'checksum, inv_v_adj_d2_2d: {np.mean(self.sol.inv_v_adj_d2_2d[0]):.8f}')
            else:
                print(f'checksum, inv_v_keep: {np.mean(self.sol.inv_v_keep[0]):.8f}')
                print(f'checksum, inv_v_adj: {np.mean(self.sol.inv_v_adj[0]):.8f}')
            return

        print('')
        for t in range(T):
            if self.par.do_2d:
                
                if t < self.par.T-1:
                    print(f'checksum, inv_w: {np.mean(self.sol.inv_w_2d[t]):.8f}')
                    print(f'checksum, q: {np.mean(self.sol.q_2d[t]):.8f}')

                print(f'checksum, inv_v_keep: {np.mean(self.sol.inv_v_keep_2d[t]):.8f}')
                print(f'checksum, inv_v_adj_full: {np.mean(self.sol.inv_v_adj_full_2d[t]):.8f}')
                print(f'checksum, inv_v_adj_d1_2d: {np.mean(self.sol.inv_v_adj_d1_2d[t]):.8f}')
                print(f'checksum, inv_v_adj_d2_2d: {np.mean(self.sol.inv_v_adj_d2_2d[t]):.8f}')
                print(f'checksum, inv_marg_u_keep: {np.mean(self.sol.inv_marg_u_keep_2d[t]):.8f}')
                print(f'checksum, inv_marg_u_adj_full: {np.mean(self.sol.inv_marg_u_adj_full_2d[t]):.8f}')
                print(f'checksum, inv_marg_u_adj_d1_2d: {np.mean(self.sol.inv_marg_u_adj_d1_2d[t]):.8f}')
                print(f'checksum, inv_marg_u_adj_d2_2d: {np.mean(self.sol.inv_marg_u_adj_d2_2d[t]):.8f}')                
                print('')

            else:

                if t < self.par.T-1:
                    print(f'checksum, inv_w: {np.mean(self.sol.inv_w[t]):.8f}')
                    print(f'checksum, q: {np.mean(self.sol.q[t]):.8f}')

                print(f'checksum, c_keep: {np.mean(self.sol.c_keep[t]):.8f}')
                print(f'checksum, d_adj: {np.mean(self.sol.d_adj[t]):.8f}')
                print(f'checksum, c_adj: {np.mean(self.sol.c_adj[t]):.8f}')
                print(f'checksum, inv_v_keep: {np.mean(self.sol.inv_v_keep[t]):.8f}')
                print(f'checksum, inv_marg_u_keep: {np.mean(self.sol.inv_marg_u_keep[t]):.8f}')
                print(f'checksum, inv_v_adj: {np.mean(self.sol.inv_v_adj[t]):.8f}')
                print(f'checksum, inv_marg_u_adj: {np.mean(self.sol.inv_marg_u_adj[t]):.8f}')
                print('')

    #########
    # solve #
    #########

    def precompile_numba(self):
        """ solve the model with very coarse grids and simulate with very few persons"""

        tic = time.time()

        # a. define
        fastpar = dict()
        fastpar['do_print'] = False
        fastpar['do_print_period'] = False
        fastpar['T'] = 2
        fastpar['Np'] = 3
        fastpar['Nn'] = 3
        fastpar['Nm'] = 3
        fastpar['Nx'] = 3
        fastpar['Na'] = 3
        fastpar['simN'] = 2
        fastpar['simT'] = 2

        # b. apply
        for key,val in fastpar.items():
            prev = getattr(self.par,key)
            setattr(self.par,key,val)
            fastpar[key] = prev

        self.setup_grids()

        # c. solve
        self.solve(do_assert=False)

        # d. simulate
        self.simulate()

        # e. reiterate
        for key,val in fastpar.items():
            setattr(self.par,key,val)
        self.setup_grids()

        toc = time.time()
        if self.par.do_print:
            print(f'numba precompiled in {toc-tic:.1f} secs')

    def solve(self,do_assert=True):
        """ solve the model
        
        Args:

            do_assert (bool,optional): make assertions on the solution
        
        """

        if self.par.do_2d:
            return self.solve_2d()

        # a. allocate solution
        keep_shape = (self.par.T,self.par.Np,self.par.Nn,self.par.Nm)
        self.sol.c_keep = np.zeros(keep_shape)
        self.sol.inv_v_keep = np.zeros(keep_shape)
        self.sol.inv_marg_u_keep = np.zeros(keep_shape)

        adj_shape = (self.par.T,self.par.Np,self.par.Nx)
        self.sol.d_adj = np.zeros(adj_shape)
        self.sol.c_adj = np.zeros(adj_shape)
        self.sol.inv_v_adj = np.zeros(adj_shape)
        self.sol.inv_marg_u_adj = np.zeros(adj_shape)
        
        post_shape = (self.par.T-1,self.par.Np,self.par.Nn,self.par.Na)
        self.sol.inv_w = np.nan*np.zeros(post_shape)
        self.sol.q = np.nan*np.zeros((self.par.T-1,self.par.Np,self.par.Nn,self.par.Na))
        self.sol.q_c = np.nan*np.zeros((self.par.T-1,self.par.Np,self.par.Nn,self.par.Na))
        self.sol.q_m = np.nan*np.zeros((self.par.T-1,self.par.Np,self.par.Nn,self.par.Na))
        
        # b. compile if using cpp
        if self.solmethod in ['vfi_cpp']:
            self.setup_cpp(use_nlopt=True)
            self.link_cpp('vfi',['solve_keep','solve_adj'])
        elif self.solmethod in ['nvfi_cpp']:
            self.setup_cpp(use_nlopt=True)
            self.link_cpp('nvfi',['compute_wq','solve_keep','solve_adj'])
        elif self.solmethod in ['negm_cpp']:
            self.setup_cpp(use_nlopt=True)
            self.link_cpp('nvfi',['solve_adj'])
            self.link_cpp('negm',['compute_wq','solve_keep'])

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
                elif self.solmethod == 'nvfi_cpp':
                    self.call_cpp('nvfi','compute_wq')                    
                elif self.solmethod == 'negm_cpp':
                    self.call_cpp('negm','compute_wq')

                toc_w = time.time()
                self.par.time_w[t] = toc_w-tic_w
                if self.par.do_print:
                    print(f'  w computed in {toc_w-tic_w:.1f} secs')

                if do_assert and self.solmethod in ['nvfi','negm']:
                    assert np.all((self.sol.inv_w[t] > 0) & (np.isnan(self.sol.inv_w[t]) == False)), t 
                    if self.solmethod in ['negm']:                                                       
                        assert np.all((self.sol.q[t] > 0) & (np.isnan(self.sol.q[t]) == False)), t

                # oo. solve keeper problem
                tic_keep = time.time()
                
                if self.solmethod == 'vfi':
                    vfi.solve_keep(t,self.sol,self.par)
                elif self.solmethod == 'nvfi':                
                    nvfi.solve_keep(t,self.sol,self.par)
                elif self.solmethod == 'negm':
                    negm.solve_keep(t,self.sol,self.par)
                elif self.solmethod == 'vfi_cpp':
                    self.call_cpp('vfi','solve_keep')
                elif self.solmethod == 'nvfi_cpp':
                    self.call_cpp('nvfi','solve_keep')
                elif self.solmethod == 'negm_cpp':
                    self.call_cpp('negm','solve_keep')                                        

                toc_keep = time.time()
                self.par.time_keep[t] = toc_keep-tic_keep
                if self.par.do_print:
                    print(f'  solved keeper problem in {toc_keep-tic_keep:.1f} secs')

                if do_assert:
                    assert np.all((self.sol.c_keep[t] >= 0) & (np.isnan(self.sol.c_keep[t]) == False)), t
                    assert np.all((self.sol.inv_v_keep[t] >= 0) & (np.isnan(self.sol.inv_v_keep[t]) == False)), t

                # ooo. solve adjuster problem
                tic_adj = time.time()
                
                if self.solmethod == 'vfi':
                    vfi.solve_adj(t,self.sol,self.par)
                elif self.solmethod in ['nvfi','negm']:
                    nvfi.solve_adj(t,self.sol,self.par)                  
                elif self.solmethod == 'vfi_cpp':
                    self.call_cpp('vfi','solve_adj')  
                elif self.solmethod in ['nvfi_cpp','negm_cpp']:
                    self.call_cpp('nvfi','solve_adj')  

                toc_adj = time.time()
                self.par.time_adj[t] = toc_adj-tic_adj
                if self.par.do_print:
                    print(f'  solved adjuster problem in {toc_adj-tic_adj:.1f} secs')

                if do_assert:
                    assert np.all((self.sol.d_adj[t] >= 0) & (np.isnan(self.sol.d_adj[t]) == False)), t
                    assert np.all((self.sol.c_adj[t] >= 0) & (np.isnan(self.sol.c_adj[t]) == False)), t
                    assert np.all((self.sol.inv_v_adj[t] >= 0) & (np.isnan(self.sol.inv_v_adj[t]) == False)), t

            # iii. print
            toc = time.time()
            if self.par.do_print or self.par.do_print_period:
                print(f' t = {t} solved in {toc-tic:.1f} secs')
        
        # d. compile if using cpp
        if self.solmethod in ['vfi_cpp']:
            self.delink_cpp('vfi')
        elif self.solmethod in ['nvfi_cpp']:
            self.delink_cpp('nvfi')
        elif self.solmethod in ['negm_cpp']:
            self.delink_cpp('nvfi')
            self.delink_cpp('negm')

    def solve_2d(self):
        """ solve the model
        
        Args:

            do_assert (bool,optional): make assertions on the solution
        
        """

        # a. allocate solution
        keep_shape = (self.par.T,self.par.Np,self.par.Nn,self.par.Nn,self.par.Nm)
        self.sol.c_keep_2d = np.zeros(keep_shape)
        self.sol.inv_v_keep_2d = np.zeros(keep_shape)
        self.sol.inv_marg_u_keep_2d = np.zeros(keep_shape)

        adj_full_shape = (self.par.T,self.par.Np,self.par.Nx)
        self.sol.d1_adj_full_2d = np.zeros(adj_full_shape)
        self.sol.d2_adj_full_2d = np.zeros(adj_full_shape)
        self.sol.c_adj_full_2d = np.zeros(adj_full_shape)
        self.sol.inv_v_adj_full_2d = np.zeros(adj_full_shape)
        self.sol.inv_marg_u_adj_full_2d = np.zeros(adj_full_shape)
        
        adj_part_shape = (self.par.T,self.par.Np,self.par.Nn,self.par.Nx)
        self.sol.d1_adj_d1_2d = np.zeros(adj_part_shape)
        self.sol.c_adj_d1_2d = np.zeros(adj_part_shape)
        self.sol.inv_v_adj_d1_2d = np.zeros(adj_part_shape)
        self.sol.inv_marg_u_adj_d1_2d = np.zeros(adj_part_shape)

        self.sol.d2_adj_d2_2d = np.zeros(adj_part_shape)
        self.sol.c_adj_d2_2d = np.zeros(adj_part_shape)
        self.sol.inv_v_adj_d2_2d = np.zeros(adj_part_shape)
        self.sol.inv_marg_u_adj_d2_2d = np.zeros(adj_part_shape)

        post_shape = (self.par.T-1,self.par.Np,self.par.Nn,self.par.Nn,self.par.Na)
        self.sol.inv_w_2d = np.nan*np.zeros(post_shape)
        self.sol.q_2d = np.nan*np.zeros((self.par.T-1,self.par.Np,self.par.Nn,self.par.Nn,self.par.Na))
        self.sol.q_c_2d = np.nan*np.zeros((self.par.T-1,self.par.Np,self.par.Nn,self.par.Nn,self.par.Na))
        self.sol.q_m_2d = np.nan*np.zeros((self.par.T-1,self.par.Np,self.par.Nn,self.par.Nn,self.par.Na))
        
        # b. compile if using cpp
        if self.solmethod in ['vfi_2d_cpp']:
            self.setup_cpp(use_nlopt=True)
            self.link_cpp('vfi_2d',['solve_keep','solve_adj_full','solve_adj_d1','solve_adj_d2'])
        elif self.solmethod in ['nvfi_2d_cpp']:
            self.setup_cpp(use_nlopt=True)
            self.link_cpp('nvfi_2d',['compute_wq','solve_keep','solve_adj_full','solve_adj_d1','solve_adj_d2'])
        elif self.solmethod in ['negm_2d_cpp']:
            self.setup_cpp(use_nlopt=True)
            self.link_cpp('nvfi_2d',['solve_adj_full','solve_adj_d1','solve_adj_d2'])
            self.link_cpp('negm_2d',['compute_wq','solve_keep'])

        # c. backwards induction
        for t in reversed(range(self.par.T)):
            
            self.par.t = t
            tic = time.time()
            
            # i. last period
            if t == self.par.T-1:

                last_period.solve_2d(t,self.sol,self.par)

            # ii. all other periods
            else:
                
                # o. compute post-decision functions
                tic_w = time.time()

                if self.solmethod == 'nvfi_2d_cpp':
                    self.call_cpp('nvfi_2d','compute_wq')                    
                elif self.solmethod == 'negm_2d_cpp':
                    self.call_cpp('negm_2d','compute_wq')

                toc_w = time.time()
                self.par.time_w[t] = toc_w-tic_w
                if self.par.do_print:
                    print(f'  w computed in {toc_w-tic_w:.1f} secs')

                # oo. solve keeper problem
                tic_keep = time.time()
                
                if self.solmethod == 'vfi_2d_cpp':
                    self.call_cpp('vfi_2d','solve_keep')
                elif self.solmethod == 'nvfi_2d_cpp':
                    self.call_cpp('nvfi_2d','solve_keep')
                elif self.solmethod == 'negm_2d_cpp':
                    self.call_cpp('negm_2d','solve_keep')                                        

                toc_keep = time.time()
                self.par.time_keep[t] = toc_keep-tic_keep
                if self.par.do_print:
                    print(f'  solved keeper problem in {toc_keep-tic_keep:.1f} secs')

                # ooo. solve adjuster problems
                tic_adj = time.time()
                
                if self.solmethod == 'vfi_2d_cpp':

                    tic_adj_full = time.time()
                    self.call_cpp('vfi_2d','solve_adj_full') 
                    self.par.time_adj_full[t] = time.time()-tic_adj_full
                    if self.par.do_print:
                        print(f'  solved full adjuster problem {self.par.time_adj_full[t]:.1f} secs')
                    
                    tic_adj_d1 = time.time()
                    self.call_cpp('vfi_2d','solve_adj_d1') 
                    self.par.time_adj_d1[t] = time.time() - tic_adj_d1
                    if self.par.do_print:
                        print(f'  solved adjuster problem with free d1 {self.par.time_adj_d1[t]:.1f} secs')

                    tic_adj_d2 = time.time()
                    self.call_cpp('vfi_2d','solve_adj_d2') 
                    self.par.time_adj_d2[t] = time.time() - tic_adj_d2
                    if self.par.do_print:
                        print(f'  solved adjuster problem with free d2 {self.par.time_adj_d2[t]:.1f} secs')

                elif self.solmethod in ['nvfi_2d_cpp','negm_2d_cpp']:

                    tic_adj_full = time.time()
                    self.call_cpp('nvfi_2d','solve_adj_full') 
                    self.par.time_adj_full[t] = time.time()-tic_adj_full
                    if self.par.do_print:
                        print(f'  solved full adjuster problem {self.par.time_adj_full[t]:.1f} secs')
                    
                    tic_adj_d1 = time.time()
                    self.call_cpp('nvfi_2d','solve_adj_d1') 
                    self.par.time_adj_d1[t] = time.time() - tic_adj_d1
                    if self.par.do_print:
                        print(f'  solved adjuster problem with free d1 {self.par.time_adj_d1[t]:.1f} secs')

                    tic_adj_d2 = time.time()
                    self.call_cpp('nvfi_2d','solve_adj_d2') 
                    self.par.time_adj_d2[t] = time.time() - tic_adj_d2
                    if self.par.do_print:
                        print(f'  solved adjuster problem with free d2 {self.par.time_adj_d2[t]:.1f} secs')

                toc_adj = time.time()
                self.par.time_adj[t] = toc_adj-tic_adj
                if self.par.do_print:
                    print(f'  solved adjuster problems in {toc_adj-tic_adj:.1f} secs')

            # iii. print
            toc = time.time()
            if self.par.do_print or self.par.do_print_period:
                print(f' t = {t} solved in {toc-tic:.1f} secs')
        
        # d. compile if using cpp
        if self.solmethod in ['vfi_2d_cpp']:
            self.delink_cpp('vfi_2d')
        elif self.solmethod in ['nvfi_2d_cpp']:
            self.delink_cpp('nvfi_2d')
        elif self.solmethod in ['negm_2d_cpp']:
            self.delink_cpp('nvfi_2d')
            self.delink_cpp('negm_2d')

    ############
    # simulate #
    ############

    def simulate(self,do_utility=False,do_euler_error=False):
        """ simulate the model """

        tic = time.time()

        # a. allocate
        self.sim.p0 = np.zeros(self.par.simN)
        if self.par.do_2d:
            self.sim.d10 = np.zeros(self.par.simN)
            self.sim.d20 = np.zeros(self.par.simN)
        else:
            self.sim.d0 = np.zeros(self.par.simN)
        self.sim.a0 = np.zeros(self.par.simN)
        self.sim.utility = np.zeros(self.par.simN)

        sim_shape = (self.par.T,self.par.simN)
        self.sim.p = np.zeros(sim_shape)
        self.sim.m = np.zeros(sim_shape)
        if self.par.do_2d:
            self.sim.n1 = np.zeros(sim_shape)
            self.sim.n2 = np.zeros(sim_shape)
        else:
            self.sim.n = np.zeros(sim_shape)
        self.sim.discrete = np.zeros(sim_shape,dtype=np.int)
        if self.par.do_2d:
            self.sim.d1 = np.zeros(sim_shape)
            self.sim.d2 = np.zeros(sim_shape)
        else:
            self.sim.d = np.zeros(sim_shape)
        self.sim.c = np.zeros(sim_shape)
        self.sim.a = np.zeros(sim_shape)
        
        euler_shape = (self.par.T-1,self.par.simN)
        self.sim.euler_error = np.zeros(euler_shape)
        self.sim.euler_error_c = np.zeros(euler_shape)

        # b. random shocks
        self.sim.p0 = np.random.lognormal(mean=0,sigma=self.par.sigma_p0,size=self.par.simN)
        if self.par.do_2d:
            self.sim.d10 = self.par.mu_d0/2*np.random.lognormal(mean=0,sigma=self.par.sigma_d0,size=self.par.simN)
            self.sim.d20 = self.par.mu_d0/2*np.random.lognormal(mean=0,sigma=self.par.sigma_d0,size=self.par.simN)
        else:
            self.sim.d0 = self.par.mu_d0*np.random.lognormal(mean=0,sigma=self.par.sigma_d0,size=self.par.simN)
        self.sim.a0 = self.par.mu_a0*np.random.lognormal(mean=0,sigma=self.par.sigma_a0,size=self.par.simN)

        I = np.random.choice(self.par.Nshocks,
            size=(self.par.T,self.par.simN), 
            p=self.par.psi_w*self.par.xi_w)
        self.sim.psi = self.par.psi[I]
        self.sim.xi = self.par.xi[I]

        # c. call
        self.par.simT = self.par.T
        simulate.lifecycle(self.sim,self.sol,self.par)

        toc = time.time()
        
        if self.par.do_print:
            print(f'model simulated in {toc-tic:.1f} secs')

        # d. euler errors
        def norm_euler_errors(model):
            return np.log10(abs(model.sim.euler_error/model.sim.euler_error_c)+1e-8)

        tic = time.time()        
        if do_euler_error:
            simulate.euler_errors(self.sim,self.sol,self.par)
            self.sim.euler_error_rel = norm_euler_errors(self)
        
        toc = time.time()
        if self.par.do_print:
            print(f'euler errors calculated in {toc-tic:.1f} secs')

        # e. utility
        tic = time.time()        
        if do_utility:
            simulate.calc_utility(self.sim,self.sol,self.par)
        
        toc = time.time()
        if self.par.do_print:
            print(f'utility calculated in {toc-tic:.1f} secs')

    ########
    # figs #
    ########

    def decision_functions(self):
        figs.decision_functions(self)

    def egm(self):        
        figs.egm(self)

    def lifecycle(self):        
        figs.lifecycle(self)

    ###########
    # analyze #
    ###########

    def analyze(self,solve=True,do_assert=True,**kwargs):

        for key,val in kwargs.items():
            setattr(self.par,key,val)
        self.setup_grids()

        # solve and simulate
        if solve:
            self.precompile_numba()
            self.solve(do_assert)
        self.simulate(do_euler_error=True,do_utility=True)

        # print
        self.print_analysis()

    def print_analysis(self):

        def avg_euler_error(model,I):
            if I.any():
                return np.nanmean(model.sim.euler_error_rel.ravel()[I])
            else:
                return np.nan

        def percentile_euler_error(model,I,p):
            if I.any():
                return np.nanpercentile(model.sim.euler_error_rel.ravel()[I],p)
            else:
                return np.nan

        # population
        keepers = self.sim.discrete[:-1,:].ravel() == 0
        adjusters = self.sim.discrete[:-1,:].ravel() > 0
        adjusters_both = self.sim.discrete[:-1,:].ravel() == 1
        adjusters_d1 = self.sim.discrete[:-1,:].ravel() == 2
        adjusters_d2 = self.sim.discrete[:-1,:].ravel() == 3
        everybody = keepers | adjusters

        # print
        time = self.par.time_w+self.par.time_adj+self.par.time_keep
        txt = f'Name: {self.name} (solmethod = {self.solmethod})\n'
        txt += f'Grids: (p,n,m,x,a) = ({self.par.Np},{self.par.Nn},{self.par.Nm},{self.par.Nx},{self.par.Na})\n'
        
        txt += 'Timings:\n'
        txt += f' total: {np.sum(time):.1f}\n'
        txt += f'     w: {np.sum(self.par.time_w):.1f}\n'
        txt += f'  keep: {np.sum(self.par.time_keep):.1f}\n'
        txt += f'   adj: {np.sum(self.par.time_adj):.1f}\n'
        txt += f'Utility: {np.mean(self.sim.utility):.6f}\n'
        
        txt += 'Euler errors:\n'
        txt += f'     total: {avg_euler_error(self,everybody):.2f} ({percentile_euler_error(self,everybody,10):.2f},{percentile_euler_error(self,everybody,90):.2f})\n'
        txt += f'   keepers: {avg_euler_error(self,keepers):.2f} ({percentile_euler_error(self,keepers,10):.2f},{percentile_euler_error(self,keepers,90):.2f})\n'
        txt += f' adjusters: {avg_euler_error(self,adjusters):.2f} ({percentile_euler_error(self,adjusters,10):.2f},{percentile_euler_error(self,adjusters,90):.2f})\n'
        
        if self.par.do_2d:
            txt += f'      both: {avg_euler_error(self,adjusters_both):.2f} ({percentile_euler_error(self,adjusters_both,10):.2f},{percentile_euler_error(self,adjusters_both,90):.2f})\n'
            txt += f'        d1: {avg_euler_error(self,adjusters_d1):.2f} ({percentile_euler_error(self,adjusters_d1,10):.2f},{percentile_euler_error(self,adjusters_d1,90):.2f})\n'
            txt += f'        d2: {avg_euler_error(self,adjusters_d2):.2f} ({percentile_euler_error(self,adjusters_d2,10):.2f},{percentile_euler_error(self,adjusters_d2,90):.2f})\n'

        txt += 'Moments:\n'
        if self.par.do_2d:
            txt += f' adjuster share: {np.mean(self.sim.discrete > 0):.3f}\n'
            txt += f'           both: {np.mean(self.sim.discrete == 1):.3f}\n'
            txt += f'        only d1: {np.mean(self.sim.discrete == 2):.3f}\n'
            txt += f'        only d2: {np.mean(self.sim.discrete == 3):.3f}\n'
        else:
            txt += f' adjuster share: {np.mean(self.sim.discrete):.3f}\n'
        
        txt += f'         mean c: {np.mean(self.sim.c):.3f}\n'
        txt += f'          var c: {np.var(self.sim.c):.3f}\n'

        if self.par.do_2d:
            txt += f'         mean d1: {np.mean(self.sim.d1):.3f}\n'
            txt += f'          var d1: {np.var(self.sim.d1):.3f}\n'
            txt += f'         mean d2: {np.mean(self.sim.d2):.3f}\n'
            txt += f'          var d2: {np.var(self.sim.d2):.3f}\n'
        else:
            txt += f'         mean d: {np.mean(self.sim.d):.3f}\n'
            txt += f'          var d: {np.var(self.sim.d):.3f}\n'

        print(txt)