# -*- coding: utf-8 -*-
"""BufferStockModel

Solves the Deaton-Carroll buffer-stock consumption model with either:

A. vfi: standard value function iteration
B. nvfi: nested value function iteration
C. egm: endogenous grid point method (egm_cpp is in C++)

"""

##############
# 1. imports #
##############

import yaml
yaml.warnings({'YAMLLoadWarning': False})

import time
import numpy as np
from numba import boolean, int32, double

# consav package
from consav import linear_interp # for linear interpolation
from consav import golden_section_search # for optimization in 1D
from consav import misc # various tools
from consav import ModelClass # baseline model class

# local modules
import utility
import last_period
import post_decision
import vfi
import nvfi
import egm
import simulate
import figs

############
# 2. model #
############

class BufferStockModelClass(ModelClass):
    
    #########
    # setup #
    #########
    
    def __init__(self,name='baseline',load=False,solmethod='vfi',compiler='vs',**kwargs):
        """ basic setup

        Args:

            name (str,optional): name, used when saving/loading
            load (bool,optinal): load from disc
            solmethod (str,optional): solmethod, used when solving
            compiler (str,optional): compiler, 'vs' or 'intel' (used for C++)
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
        parlist = [ # (name,numba type), parameters, grids etc.
            ('T',int32), # integer 32bit
            ('beta',double), # double
            ('rho',double),
            ('R',double),
            ('sigma_psi',double),
            ('Npsi',int32),
            ('sigma_xi',double),
            ('Nxi',int32),
            ('pi',double),
            ('mu',double),
            ('Nm',int32),
            ('grid_m',double[:]), # 1d array of doubles
            ('Np',int32),
            ('grid_p',double[:]),    
            ('Na',int32),
            ('grid_a',double[:]),        
            ('Nshocks',int32),        
            ('psi',double[:]),        
            ('psi_w',double[:]),        
            ('xi',double[:]),        
            ('xi_w',double[:]),        
            ('tol',double),
            ('simT',int32), 
            ('simN',int32), 
            ('sim_seed',int32),
            ('do_print',boolean), # boolean
            ('do_simple_w',boolean),
            ('cppthreads',int32) 
        ]
        
        sollist = [ # (name, numba type), solution data
            ('v',double[:,:,:]), # 3d array of doubles
            ('c',double[:,:,:]),
            ('w',double[:,:]), # 2d array of doubles
            ('q',double[:,:]),
        ]        

        simlist = [ # (name, numba type), simulation data
            ('p',double[:,:]),
            ('m',double[:,:]),
            ('c',double[:,:]),
            ('a',double[:,:]),
            ('xi',double[:,:]),
            ('psi',double[:,:])
        ]      

        # b. create subclasses
        self.par,self.sol,self.sim = self.create_subclasses(parlist,sollist,simlist)

        # note: the above returned classes are in a format where they can be used in numba functions

        # c. load
        if load:
            self.load()
        else:
            self.setup(**kwargs)

    def setup(self,**kwargs):
        """ define baseline values and update with user choices

        Args:

             **kwargs: change to baseline parameters in .par

        """   

        # a. baseline parameters
        
        # horizon
        self.par.T = 5
        
        # preferences
        self.par.beta = 0.96
        self.par.rho = 2

        # returns and income
        self.par.R = 1.03
        self.par.sigma_psi = 0.1
        self.par.Npsi = 6
        self.par.sigma_xi = 0.1
        self.par.Nxi = 6
        self.par.pi = 0.1
        self.par.mu = 0.5
        
        # grids (number of points)
        self.par.Nm = 600
        self.par.Np = 400
        self.par.Na = 800

        # misc
        self.par.tol = 1e-8
        self.par.do_print = True
        self.par.do_simple_w = False
        self.par.cppthreads = 1

        # simulation
        self.par.simT = self.par.T
        self.par.simN = 1000
        self.par.sim_seed = 1998

        # b. update baseline parameters using keywords 
        for key,val in kwargs.items():
            setattr(self.par,key,val) # like par.key = val
        
        # c. setup_grids
        self.setup_grids()
        
    def setup_grids(self):
        """ construct grids for states and shocks """

        # a. states (unequally spaced vectors of length Nm)
        self.par.grid_m = misc.nonlinspace(1e-6,20,self.par.Nm,1.1)
        self.par.grid_p = misc.nonlinspace(1e-4,10,self.par.Np,1.1)
        
        # b. post-decision states (unequally spaced vector of length Na)
        self.par.grid_a = misc.nonlinspace(1e-6,20,self.par.Na,1.1)
        
        # c. shocks (qudrature nodes and weights using GaussHermite)
        shocks = misc.create_shocks(
            self.par.sigma_psi,self.par.Npsi,self.par.sigma_xi,self.par.Nxi,
            self.par.pi,self.par.mu)
        self.par.psi,self.par.psi_w,self.par.xi,self.par.xi_w,self.par.Nshocks = shocks

        # d. set seed
        np.random.seed(self.par.sim_seed)

    def checksum(self):
        """ print checksum """

        print(f'checksum: {np.mean(self.sol.c[0])}')

    #########
    # solve #
    #########

    def _solve_prep(self):
        """ allocate memory for solution """

        self.sol.c = np.nan*np.ones((self.par.T,self.par.Np,self.par.Nm))        
        self.sol.v = np.nan*np.zeros((self.par.T,self.par.Np,self.par.Nm))
        self.sol.w = np.nan*np.zeros((self.par.Np,self.par.Na))
        self.sol.q = np.nan*np.zeros((self.par.Np,self.par.Na))

    def solve(self):
        """ solve the model using solmethod """

        # a. allocate solution
        self._solve_prep()
        
        # b. backwards induction
        for t in reversed(range(self.par.T)):
            
            tic = time.time()
            
            # i. last period
            if t == self.par.T-1:
                
                last_period.solve(t,self.sol,self.par)

            # ii. all other periods
            else:
                
                # o. compute post-decision functions
                tic_w = time.time()

                compute_w,compute_q = False,False
                if self.solmethod in ['nvfi']:
                    compute_w=True
                elif self.solmethod in ['egm']:
                    compute_q=True
                if compute_w or compute_q:
                    if self.par.do_simple_w:
                        post_decision.compute_wq_simple(t,self.sol,self.par,compute_w=compute_w,compute_q=compute_q)
                    else:
                        post_decision.compute_wq(t,self.sol,self.par,compute_w=compute_w,compute_q=compute_q)

                toc_w = time.time()

                # oo. solve bellman equation
                if self.solmethod == 'vfi':
                    vfi.solve_bellman(t,self.sol,self.par)                    
                elif self.solmethod == 'nvfi':
                    nvfi.solve_bellman(t,self.sol,self.par)
                elif self.solmethod == 'egm':
                    egm.solve_bellman(t,self.sol,self.par)                    
                else:
                    raise ValueError(f'unknown solution method, {self.solmethod}')

            # iii. print
            toc = time.time()
            if self.par.do_print:
                msg = f' t = {t} solved in {toc-tic:.1f} secs'
                if t < self.par.T-1:
                    msg += f' (w: {toc_w-tic_w:.1f} secs)'                
                print(msg)

    def solve_cpp(self,compiler='vs'):
        """ solve the model using egm written in C++
        
        Args:
            compiler (str,optional): compiler choice (vs or intel)

        """

        EGM = 'EGM'
        
        # a. allocate solution
        self._solve_prep()

        # b. compile
        funcnames = ['solve','simulate']
        self.setup_cpp()
        self.link_cpp(EGM,funcnames)

        # c. solve by EGM
        tic = time.time()
       
        if self.solmethod in ['egm']:
            self.call_cpp(EGM,'solve')
        else:
            raise ValueError(f'unknown cpp solution method, {self.solmethod}')            
        
        toc = time.time()

        # d. delink
        self.delink_cpp(EGM)

        return tic,toc

    ############
    # simulate #
    ############

    def _simulate_prep(self):
        """ allocate memory for simulation and draw random numbers """

        # a. allocate
        self.sim.p = np.nan*np.zeros((self.par.simT,self.par.simN))
        self.sim.m = np.nan*np.zeros((self.par.simT,self.par.simN))
        self.sim.c = np.nan*np.zeros((self.par.simT,self.par.simN))
        self.sim.a = np.nan*np.zeros((self.par.simT,self.par.simN))

        # b. draw random shocks
        I = np.random.choice(self.par.Nshocks,
            size=(self.par.T,self.par.simN), 
            p=self.par.psi_w*self.par.xi_w)
        self.sim.psi = self.par.psi[I]
        self.sim.xi = self.par.xi[I]

    def simulate(self):
        """ simulate model """

        tic = time.time()

        # a. allocate memory and draw random numbers
        self._simulate_prep()

        # b. simulate
        self.par.simT = self.par.T
        simulate.lifecycle(self.sim,self.sol,self.par)

        toc = time.time()

        if self.par.do_print:
            print(f'model simulated in {toc-tic:.1f} secs')

    ########
    # figs #
    ########

    def consumption_function(self,t=0):
        figs.consumption_function(self,t)

    def consumption_function_interact(self):
        figs.consumption_function_interact(self)
          
    def lifecycle(self):
        figs.lifecycle(self)
    
    ########
    # figs #
    ########

    def test(self):
        """ method for specifying test """
        
        # a. save print status
        do_print = self.par.do_print
        self.par.do_print = False

        # b. test run
        self.solve()

        # c. timed run
        tic = time.time()
        self.solve()
        toc = time.time()
        print(f'solution time: {toc-tic:.1f} secs')
        self.checksum()

        # d. reset print status
        self.par.do_print = do_print