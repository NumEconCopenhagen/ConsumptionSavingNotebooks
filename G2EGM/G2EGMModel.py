# -*- coding: utf-8 -*-
"""G2EGModel

"""

##############
# 1. imports #
##############

import time
import numpy as np
from numba import boolean, int64, double

# consav package
from consav import misc # various functions
from consav import ModelClass # baseline model classes

# local modules
import pens
import utility
import retirement
import last_period
import post_decision
import G2EGM
import NEGM
import simulate

class G2EGMModelClass(ModelClass):
    
    #########
    # setup #
    #########

    def __init__(self,name='baseline',load=False,solmethod='G2EGM',compiler='vs',**kwargs):
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

            ('T',int64),

            ('beta',double),
            ('rho',double),
            ('alpha',double),
            
            ('yret',double),
            ('var_eta',double),
            ('Ra',double),
            ('Rb',double),
            ('chi',double),

            ('phi_m',double),
            ('phi_n',double),

            ('Nm_ret',int64),
            ('m_max_ret',double),
            ('grid_m_ret',double[:]),
            ('Na_ret',int64),
            ('a_max_ret',double),
            ('grid_a_ret',double[:]),
            ('Nmcon_ret',int64),

            ('Nm',int64),
            ('m_max',int64),
            ('grid_m',double[:]),
            ('grid_m_nd',double[:,:]),

            ('n_add',double),
            ('Nn',int64),
            ('n_max',int64),
            ('grid_n',double[:]),
            ('grid_n_nd',double[:,:]),

            ('pd_fac',double),
            ('a_add',double),
            ('b_add',double),
            ('Na_pd',int64),
            ('a_max',double),
            ('grid_a_pd',double[:]),
            ('grid_a_pd_nd',double[:,:]),
            ('Nb_pd',int64),
            ('b_max',double),
            ('grid_b_pd',double[:]),            
            ('grid_b_pd_nd',double[:,:]),            

            ('d_dcon',double[:,:]),
            ('acon_fac',double),
            ('Nc_acon',int64),
            ('Nb_acon',int64),
            ('grid_b_acon',double[:]),
            ('b_acon',double[:]),
            ('a_acon',double[:]),
            ('con_fac',double),
            ('Nc_con',int64),
            ('Nb_con',int64),
            ('grid_c_con',double[:]),
            ('grid_b_con',double[:]),
            ('c_con',double[:,:]),
            ('b_con',double[:,:]),
            ('a_con',double[:,:]),
            ('d_con',double[:,:]),
            
            ('Neta',int64),
            ('eta',double[:]),
            ('w_eta',double[:]),

            ('eulerK',int64),

            ('egm_extrap_add',int64),
            ('egm_extrap_w',double),
            ('delta_con',double),
            ('eps',double),
            ('do_print',boolean),

            ('grid_l',double[:]),

            ('time_work',double[:]),
            ('time_w',double[:]),
            ('time_egm',double[:]),
            ('time_vfi',double[:]),

        ]

        sollist = [
            ('m_ret',double[:,:]),
            ('c_ret',double[:,:]),
            ('a_ret',double[:,:]),
            ('v_ret',double[:,:]),
            ('inv_v_ret',double[:,:]),
            ('inv_vm_ret',double[:,:]),
            ('inv_vn_ret',double[:,:]),
            ('c',double[:,:,:]),
            ('d',double[:,:,:]),
            ('inv_v',double[:,:,:]),
            ('inv_vm',double[:,:,:]),
            ('inv_vn',double[:,:,:]),
            ('w',double[:,:,:]),
            ('wa',double[:,:,:]),
            ('wb',double[:,:,:]),
            ('z',double[:,:,:]),
            ('ucon_c',double[:,:,:]),
            ('ucon_d',double[:,:,:]),
            ('ucon_v',double[:,:,:]),
            ('dcon_c',double[:,:,:]),
            ('dcon_d',double[:,:,:]),
            ('dcon_v',double[:,:,:]),            
            ('acon_c',double[:,:,:]),
            ('acon_d',double[:,:,:]),
            ('acon_v',double[:,:,:]),            
            ('con_c',double[:,:,:]),
            ('con_d',double[:,:,:]),
            ('con_v',double[:,:,:]),
            ('c_pure_c',double[:,:,:]),
            ('inv_v_pure_c',double[:,:,:])
        ]

        simlist = [
            ('euler',double[:,:,:]),
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
        self.par.T = 20
        
        # preferences
        self.par.beta = 0.98
        self.par.rho = 2
        self.par.alpha = 0.25

        # returns and income
        self.par.yret = 0.5
        self.par.var_eta = 0
        self.par.Ra = 1.02
        self.par.Rb = 1.04
        self.par.chi = 0.10
        self.par.Neta = 1

        # grids
        self.par.Nm_ret = 500
        self.par.m_max_ret = 50.0
        self.par.Na_ret = 400
        self.par.a_max_ret = 25.0
        self.par.Nm = 600
        self.par.m_max = 10.0    
        self.par.phi_m = 1.1  
        self.par.n_add = 2.00
        self.par.phi_n = 1.25  
        self.par.acon_fac = 0.25
        self.par.con_fac = 0.50
        self.par.pd_fac = 2.00
        self.par.a_add = -2.00
        self.par.b_add = 2.00

        # euler
        self.par.eulerK = 100

        # misc
        self.par.egm_extrap_add = 2
        self.par.egm_extrap_w = -0.25
        self.par.delta_con = 0.001
        self.par.eps = 1e-6
        self.par.do_print = False

        # b. update baseline parameters using keywords 
        for key,val in kwargs.items():
            setattr(self.par,key,val) # like par.key = val

        # c. setup grids
        self.setup_grids()
        
    def setup_grids(self):
        """ construct grids for states and shocks """
        
        # a. retirement
    
        # pre-decision states
        self.par.grid_m_ret = misc.nonlinspace(self.par.eps,self.par.m_max_ret,self.par.Nm_ret,self.par.phi_m)
        self.par.Nmcon_ret = self.par.Nm_ret - self.par.Na_ret
        
        # post-decision states
        self.par.grid_a_ret = misc.nonlinspace(0,self.par.a_max_ret,self.par.Na_ret,self.par.phi_m)
        
        # b. working: state space (m,n,k)    
        self.par.grid_m = misc.nonlinspace(self.par.eps,self.par.m_max,self.par.Nm,self.par.phi_m)

        self.par.Nn = self.par.Nm
        self.par.n_max = self.par.m_max + self.par.n_add
        self.par.grid_n = misc.nonlinspace(0,self.par.n_max,self.par.Nn,self.par.phi_n)

        self.par.grid_n_nd, self.par.grid_m_nd = np.meshgrid(self.par.grid_n,self.par.grid_m,indexing='ij')

        # c. working: w interpolant (and wa and wb and wq)
        self.par.Na_pd = np.int64(np.floor(self.par.pd_fac*self.par.Nm))
        self.par.a_max = self.par.m_max + self.par.a_add
        self.par.grid_a_pd = misc.nonlinspace(0,self.par.a_max,self.par.Na_pd,self.par.phi_m)
    
        self.par.Nb_pd = np.int64(np.floor(self.par.pd_fac*self.par.Nn))
        self.par.b_max = self.par.n_max + self.par.b_add
        self.par.grid_b_pd = misc.nonlinspace(0,self.par.b_max,self.par.Nb_pd,self.par.phi_n)
    
        self.par.grid_b_pd_nd, self.par.grid_a_pd_nd = np.meshgrid(self.par.grid_b_pd,self.par.grid_a_pd,indexing='ij')
        
        # d. working: egm (seperate grids for each segment)
        
        if self.solmethod == 'G2EGM':

            # i. dcon
            self.par.d_dcon = np.zeros((self.par.Na_pd,self.par.Nb_pd),dtype=np.float_,order='C')
                
            # ii. acon
            self.par.Nc_acon = np.int64(np.floor(self.par.Na_pd*self.par.acon_fac))
            self.par.Nb_acon = np.int64(np.floor(self.par.Nb_pd*self.par.acon_fac))
            self.par.grid_b_acon = misc.nonlinspace(0,self.par.b_max,self.par.Nb_acon,self.par.phi_n)
            self.par.a_acon = np.zeros(self.par.grid_b_acon.shape)
            self.par.b_acon = self.par.grid_b_acon

            # iii. con
            self.par.Nc_con = np.int64(np.floor(self.par.Na_pd*self.par.con_fac))
            self.par.Nb_con = np.int64(np.floor(self.par.Nb_pd*self.par.con_fac))
            
            self.par.grid_c_con = misc.nonlinspace(self.par.eps,self.par.m_max,self.par.Nc_con,self.par.phi_m)
            self.par.grid_b_con = misc.nonlinspace(0,self.par.b_max,self.par.Nb_con,self.par.phi_n)

            self.par.b_con,self.par.c_con = np.meshgrid(self.par.grid_b_con,self.par.grid_c_con,indexing='ij')
            self.par.a_con = np.zeros(self.par.c_con.shape)
            self.par.d_con = np.zeros(self.par.c_con.shape)
        
        elif self.solmethod == 'NEGM':

            self.par.grid_l = self.par.grid_m

        # e. shocks
        assert (self.par.Neta == 1 and self.par.var_eta == 0) or (self.par.Neta > 1 and self.par.var_eta > 0)

        if self.par.Neta > 1:
            self.par.eta,self.par.w_eta = misc.normal_gauss_hermite(np.sqrt(self.par.var_eta), self.par.Neta)
        else:
            self.par.eta = np.ones(1)
            self.par.w_eta = np.ones(1)

        # f. timings
        self.par.time_work = np.zeros(self.par.T)
        self.par.time_w = np.zeros(self.par.T)
        self.par.time_egm = np.zeros(self.par.T)
        self.par.time_vfi = np.zeros(self.par.T)

    def solve(self):

        if self.solmethod == 'G2EGM':
            self.solve_G2EGM()
        elif self.solmethod == 'NEGM':
            self.solve_NEGM()

    def precompile_numba(self):

        t0 = time.time()

        # a. remember actual settings
        prev = dict()
        varnames = ['T','Nm','do_print','Nm_ret','Na_ret']
        for varname in varnames:
            prev[varname] = getattr(self.par,varname)

        # b. fast settings
        self.par.T = 2
        self.par.Nm_ret = 20
        self.par.Na_ret = 10
        self.par.Nm = 5
        self.par.do_print = False
        self.setup_grids()

        # c. solve
        self.solve()

        # d. reset
        for varname in varnames:
            setattr(self.par,varname,prev[varname]) 
        self.setup_grids()

        if self.par.do_print:
            print(f'pre-compiled numba in {time.time()-t0:.2f} secs')

    def solve_G2EGM(self):
        
        if self.par.do_print:
            print('Solving with G2EGM:')

        # a. allocate
        self.sol.m_ret = np.zeros((self.par.T,self.par.Nm_ret))
        self.sol.c_ret = np.zeros((self.par.T,self.par.Nm_ret))
        self.sol.a_ret = np.zeros((self.par.T,self.par.Nm_ret))
        self.sol.inv_v_ret = np.zeros((self.par.T,self.par.Nm_ret))
        self.sol.inv_vm_ret = np.zeros((self.par.T,self.par.Nm_ret))
        self.sol.inv_vn_ret = np.zeros((self.par.T,self.par.Nm_ret))

        self.sol.c = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.d = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.inv_v = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.inv_vm = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.inv_vn = np.zeros((self.par.T,self.par.Nn,self.par.Nm))

        self.sol.ucon_c = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.ucon_d = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.ucon_v = np.zeros((self.par.T,self.par.Nn,self.par.Nm))

        self.sol.dcon_c = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.dcon_d = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.dcon_v = np.zeros((self.par.T,self.par.Nn,self.par.Nm))

        self.sol.acon_c = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.acon_d = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.acon_v = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.con_c = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.con_d = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.con_v = np.zeros((self.par.T,self.par.Nn,self.par.Nm))

        self.sol.z = np.zeros((self.par.T,self.par.Nn,self.par.Nm))

        self.sol.w = np.zeros((self.par.T-1,self.par.Nb_pd,self.par.Na_pd))
        self.sol.wa = np.zeros((self.par.T-1,self.par.Nb_pd,self.par.Na_pd))
        self.sol.wb = np.zeros((self.par.T-1,self.par.Nb_pd,self.par.Na_pd))

        # b. solve retirement
        t0 = time.time()

        retirement.solve(self.sol,self.par)

        if self.par.do_print:
            print(f'solved retirement problem in {time.time()-t0:.2f} secs')

        # c. solve last period working
        t0 = time.time()

        last_period.solve(self.sol,self.par)

        if self.par.do_print:
            print(f'solved last period working in {time.time()-t0:.2f} secs')

        # d. solve working
        for t in reversed(range(self.par.T-1)):
            
            t0 = time.time()
            
            if self.par.do_print:
                print(f' t = {t}:')
            
            # i. post decision
            t0_w = time.time()

            post_decision.compute(t,self.sol,self.par)

            self.par.time_w[t] = time.time()-t0_w
            if self.par.do_print:
                print(f'   computed post decision value function in {self.par.time_w[t]:.2f} secs')

            # ii. EGM
            t0_EGM = time.time()
            
            G2EGM.solve(t,self.sol,self.par)
            
            self.par.time_egm[t] = time.time()-t0_EGM
            if self.par.do_print:
                print(f'   applied G2EGM  in {self.par.time_egm[t]:.2f} secs')

            self.par.time_work[t] = time.time()-t0

        if self.par.do_print:
            print(f'solved working problem in {np.sum(self.par.time_work):.2f} secs')

    def solve_NEGM(self):
        
        if self.par.do_print:
            print('Solving with NEGM:')

        # a. allocate
        self.sol.m_ret = np.zeros((self.par.T,self.par.Nm_ret))
        self.sol.c_ret = np.zeros((self.par.T,self.par.Nm_ret))
        self.sol.a_ret = np.zeros((self.par.T,self.par.Nm_ret))
        self.sol.inv_v_ret = np.zeros((self.par.T,self.par.Nm_ret))
        self.sol.inv_vm_ret = np.zeros((self.par.T,self.par.Nm_ret))
        self.sol.inv_vn_ret = np.zeros((self.par.T,self.par.Nm_ret))

        self.sol.c = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.d = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.inv_v = np.zeros((self.par.T,self.par.Nn,self.par.Nm))
        self.sol.inv_vm = np.zeros((self.par.T,self.par.Nn,self.par.Nm))

        self.sol.w = np.zeros((self.par.T-1,self.par.Nb_pd,self.par.Na_pd))
        self.sol.wa = np.zeros((self.par.T-1,self.par.Nb_pd,self.par.Na_pd))
        
        self.sol.c_pure_c = np.zeros((self.par.T,self.par.Nb_pd,self.par.Nm))
        self.sol.inv_v_pure_c = np.zeros((self.par.T,self.par.Nb_pd,self.par.Nm))

        # b. solve retirement
        t0 = time.time()

        retirement.solve(self.sol,self.par,G2EGM=False)

        if self.par.do_print:
            print(f'solved retirement problem in {time.time()-t0:.2f} secs')

        # c. solve last period working
        t0 = time.time()

        last_period.solve(self.sol,self.par,G2EGM=False)

        if self.par.do_print:
            print(f'solved last period working in {time.time()-t0:.2f} secs')

        # d. solve working  
        for t in reversed(range(self.par.T-1)):
            
            t0 = time.time()   
            
            if self.par.do_print:
                print(f' t = {t}:')
            
            # i. post decision
            t0_w = time.time()

            post_decision.compute(t,self.sol,self.par,G2EGM=False)

            self.par.time_w[t] = time.time() - t0_w
            if self.par.do_print:
                print(f'   computed post decision value function in {self.par.time_w[t]:.2f} secs')

            # ii. pure consumption problem
            t0_egm = time.time()
            
            NEGM.solve_pure_c(t,self.sol,self.par)
            
            self.par.time_egm[t] = time.time()-t0_egm
            if self.par.do_print:
                print(f'   solved pure consumption problem in {self.par.time_egm[t]:.2f} secs')

            # iii. outer problem
            t0_vfi = time.time()
            
            NEGM.solve_outer(t,self.sol,self.par)
            
            self.par.time_vfi[t] = time.time()-t0_vfi
            if self.par.do_print:
                print(f'   solved outer problem in {self.par.time_vfi[t] :.2f} secs')

            self.par.time_work[t] = time.time()-t0

        if self.par.do_print:
            print(f'solved working problem in {np.sum(self.par.time_work):.2f} secs')

    def checksums(self,Ts=None):

        if Ts == 0:
            Ts = list(range(self.par.T))

        print('retirement')
        for t in Ts:
            print(f't = {t}, c: {np.sum(self.sol.c_ret[t,:]):.8f}')
            print(f't = {t}, v: {np.sum(self.sol.inv_v_ret[t,:]):.8f}')
            print('')
            
        print('working')
        for t in Ts:
            print('')
            print(f't = {t}, c: {np.sum(self.sol.c[t,:,:]):.8f}')   
            if self.solmethod == 'G2EGM': 
                print(f't = {t},  ucon: {np.nansum(self.sol.ucon_c[t,:,:]):.8f}')    
                print(f't = {t},  dcon: {np.nansum(self.sol.dcon_c[t,:,:]):.8f}')    
                print(f't = {t},  acon: {np.nansum(self.sol.acon_c[t,:,:]):.8f}')    
                print(f't = {t},  con: {np.nansum(self.sol.con_c[t,:,:]):.8f}')    
            print(f't = {t}, v: {np.sum(self.sol.inv_v[t,:,:]):.8f}')     
            if self.solmethod == 'G2EGM': 
                print(f't = {t},  ucon: {np.nansum(self.sol.ucon_c[t,:,:]):.8f}')    
                print(f't = {t},  dcon: {np.nansum(self.sol.dcon_c[t,:,:]):.8f}')    
                print(f't = {t},  acon: {np.nansum(self.sol.acon_c[t,:,:]):.8f}')    
                print(f't = {t},  con: {np.nansum(self.sol.con_c[t,:,:]):.8f}')               
            print(f't = {t}, vm: {np.sum(self.sol.inv_vm[t,:,:]):.8f}')     
            if self.solmethod == 'G2EGM': 
                print(f't = {t}, vn: {np.sum(self.sol.inv_vn[t,:,:]):.8f}')  
            if t < self.par.T-1:
                print(f't = {t}, w: {np.sum(self.sol.w[t,:,:]):.8f}')    
                print(f't = {t}, wa: {np.sum(self.sol.wa[t,:,:]):.8f}')    
                if self.solmethod == 'G2EGM': 
                    print(f't = {t}, wb: {np.sum(self.sol.wb[t,:,:]):.8f}')    
    
    def calculate_euler(self):

        self.sim.euler = np.full((self.par.T-1,self.par.eulerK,self.par.eulerK),np.nan)
        simulate.euler(self.sim,self.sol,self.par)