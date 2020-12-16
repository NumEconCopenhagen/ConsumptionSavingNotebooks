# -*- coding: utf-8 -*-
"""G2EGModel

"""

##############
# 1. imports #
##############

import time
import numpy as np

# consav package
from consav import ModelClass, jit # baseline model class and jit
from consav.grids import nonlinspace # grids
from consav.quadrature import log_normal_gauss_hermite # income shocks

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

    def settings(self):

        # a. namespaces
        self.namespaces = []
        
        # b. other attributes
        self.other_attrs = []
        
        # c. savefolder
        self.savefolder = 'saved'

        # d. list not-floates for safe type inference
        self.not_floats = ['solmethod','T','Nm_ret','Na_ret','Nmcon_ret','Nm','m_max','Nn','n_max',
                           'Na_pd','Nb_pd','Nc_acon','Nb_acon','Nc_con','Nb_con','Neta',
                           'eulerK','egm_extrap_add','do_print']
                
    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. baseline parameters
        
        # horizon
        par.T = 20
        
        # preferences
        par.beta = 0.98
        par.rho = 2.0
        par.alpha = 0.25

        # returns and income
        par.yret = 0.5
        par.var_eta = 0.0
        par.Ra = 1.02
        par.Rb = 1.04
        par.chi = 0.10
        par.Neta = 1

        # grids
        par.Nm_ret = 500
        par.m_max_ret = 50.0
        par.Na_ret = 400
        par.a_max_ret = 25.0
        par.Nm = 600
        par.m_max = 10.0    
        par.phi_m = 1.1  
        par.n_add = 2.00
        par.phi_n = 1.25  
        par.acon_fac = 0.25
        par.con_fac = 0.50
        par.pd_fac = 2.00
        par.a_add = -2.00
        par.b_add = 2.00

        # euler
        par.eulerK = 100

        # misc
        par.solmethod = 'G2EGM'
        par.egm_extrap_add = 2
        par.egm_extrap_w = -0.25
        par.delta_con = 0.001
        par.eps = 1e-6
        par.do_print = False
        
    def allocate(self):
        """ allocate model, i.e. create grids and allocate solution and simluation arrays """

        # a. grid
        self.create_grids()

        # b. solution
        self.solve_prep()

        # c. simulation
        self.sim.euler = np.full((self.par.T-1,self.par.eulerK,self.par.eulerK),np.nan)

    def create_grids(self):
        """ construct grids for states and shocks """
        
        par = self.par

        # a. retirement
    
        # pre-decision states
        par.grid_m_ret = nonlinspace(par.eps,par.m_max_ret,par.Nm_ret,par.phi_m)
        par.Nmcon_ret = par.Nm_ret - par.Na_ret
        
        # post-decision states
        par.grid_a_ret = nonlinspace(0,par.a_max_ret,par.Na_ret,par.phi_m)
        
        # b. working: state space (m,n,k)    
        par.grid_m = nonlinspace(par.eps,par.m_max,par.Nm,par.phi_m)

        par.Nn = par.Nm
        par.n_max = par.m_max + par.n_add
        par.grid_n = nonlinspace(0,par.n_max,par.Nn,par.phi_n)

        par.grid_n_nd, par.grid_m_nd = np.meshgrid(par.grid_n,par.grid_m,indexing='ij')

        # c. working: w interpolant (and wa and wb and wq)
        par.Na_pd = np.int64(np.floor(par.pd_fac*par.Nm))
        par.a_max = par.m_max + par.a_add
        par.grid_a_pd = nonlinspace(0,par.a_max,par.Na_pd,par.phi_m)
    
        par.Nb_pd = np.int64(np.floor(par.pd_fac*par.Nn))
        par.b_max = par.n_max + par.b_add
        par.grid_b_pd = nonlinspace(0,par.b_max,par.Nb_pd,par.phi_n)
    
        par.grid_b_pd_nd, par.grid_a_pd_nd = np.meshgrid(par.grid_b_pd,par.grid_a_pd,indexing='ij')
        
        # d. working: egm (seperate grids for each segment)
        
        if par.solmethod == 'G2EGM':

            # i. dcon
            par.d_dcon = np.zeros((par.Na_pd,par.Nb_pd),dtype=np.float_,order='C')
                
            # ii. acon
            par.Nc_acon = np.int64(np.floor(par.Na_pd*par.acon_fac))
            par.Nb_acon = np.int64(np.floor(par.Nb_pd*par.acon_fac))
            par.grid_b_acon = nonlinspace(0,par.b_max,par.Nb_acon,par.phi_n)
            par.a_acon = np.zeros(par.grid_b_acon.shape)
            par.b_acon = par.grid_b_acon

            # iii. con
            par.Nc_con = np.int64(np.floor(par.Na_pd*par.con_fac))
            par.Nb_con = np.int64(np.floor(par.Nb_pd*par.con_fac))
            
            par.grid_c_con = nonlinspace(par.eps,par.m_max,par.Nc_con,par.phi_m)
            par.grid_b_con = nonlinspace(0,par.b_max,par.Nb_con,par.phi_n)

            par.b_con,par.c_con = np.meshgrid(par.grid_b_con,par.grid_c_con,indexing='ij')
            par.a_con = np.zeros(par.c_con.shape)
            par.d_con = np.zeros(par.c_con.shape)
        
        elif par.solmethod == 'NEGM':

            par.grid_l = par.grid_m

        # e. shocks
        assert (par.Neta == 1 and par.var_eta == 0) or (par.Neta > 1 and par.var_eta > 0)

        if par.Neta > 1:
            par.eta,par.w_eta = log_normal_gauss_hermite(np.sqrt(par.var_eta), par.Neta)
        else:
            par.eta = np.ones(1)
            par.w_eta = np.ones(1)

        # f. timings
        par.time_work = np.zeros(par.T)
        par.time_w = np.zeros(par.T)
        par.time_egm = np.zeros(par.T)
        par.time_vfi = np.zeros(par.T)

    def solve(self):
        """ solve the model """

        if self.par.solmethod == 'G2EGM':
            self.solve_G2EGM()
        elif self.par.solmethod == 'NEGM':
            self.solve_NEGM()

    def precompile_numba(self):
        """ solve the model with very coarse grids"""

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
        self.allocate()

        # c. solve
        self.solve()

        # d. reset
        for varname in varnames:
            setattr(self.par,varname,prev[varname]) 

        self.allocate()

        if self.par.do_print:
            print(f'pre-compiled numba in {time.time()-t0:.2f} secs')

    def solve_prep(self):
        """ allocate memory for solution """

        par = self.par
        sol = self.sol

        # a. retirement
        sol.m_ret = np.zeros((par.T,par.Nm_ret))
        sol.c_ret = np.zeros((par.T,par.Nm_ret))
        sol.a_ret = np.zeros((par.T,par.Nm_ret))
        sol.inv_v_ret = np.zeros((par.T,par.Nm_ret))
        sol.inv_vm_ret = np.zeros((par.T,par.Nm_ret))
        sol.inv_vn_ret = np.zeros((par.T,par.Nm_ret))

        # b. working
        if par.solmethod == 'G2EGM':

            sol.c = np.zeros((par.T,par.Nn,par.Nm))
            sol.d = np.zeros((par.T,par.Nn,par.Nm))
            sol.inv_v = np.zeros((par.T,par.Nn,par.Nm))
            sol.inv_vm = np.zeros((par.T,par.Nn,par.Nm))
            sol.inv_vn = np.zeros((par.T,par.Nn,par.Nm))

            sol.ucon_c = np.zeros((par.T,par.Nn,par.Nm))
            sol.ucon_d = np.zeros((par.T,par.Nn,par.Nm))
            sol.ucon_v = np.zeros((par.T,par.Nn,par.Nm))

            sol.dcon_c = np.zeros((par.T,par.Nn,par.Nm))
            sol.dcon_d = np.zeros((par.T,par.Nn,par.Nm))
            sol.dcon_v = np.zeros((par.T,par.Nn,par.Nm))

            sol.acon_c = np.zeros((par.T,par.Nn,par.Nm))
            sol.acon_d = np.zeros((par.T,par.Nn,par.Nm))
            sol.acon_v = np.zeros((par.T,par.Nn,par.Nm))
            sol.con_c = np.zeros((par.T,par.Nn,par.Nm))
            sol.con_d = np.zeros((par.T,par.Nn,par.Nm))
            sol.con_v = np.zeros((par.T,par.Nn,par.Nm))

            sol.z = np.zeros((par.T,par.Nn,par.Nm))

            sol.w = np.zeros((par.T-1,par.Nb_pd,par.Na_pd))
            sol.wa = np.zeros((par.T-1,par.Nb_pd,par.Na_pd))
            sol.wb = np.zeros((par.T-1,par.Nb_pd,par.Na_pd))
            
        elif par.solmethod == 'NEGM':

            sol.c = np.zeros((par.T,par.Nn,par.Nm))
            sol.d = np.zeros((par.T,par.Nn,par.Nm))
            sol.inv_v = np.zeros((par.T,par.Nn,par.Nm))
            sol.inv_vn = np.zeros((0,0,0))
            sol.inv_vm = np.zeros((par.T,par.Nn,par.Nm))

            sol.w = np.zeros((par.T-1,par.Nb_pd,par.Na_pd))
            sol.wa = np.zeros((par.T-1,par.Nb_pd,par.Na_pd))
            sol.wb = np.zeros((0,0,0))
            
            sol.c_pure_c = np.zeros((par.T,par.Nb_pd,par.Nm))
            sol.inv_v_pure_c = np.zeros((par.T,par.Nb_pd,par.Nm))
            
    def solve_G2EGM(self):
        """ solve with G2EGM """
        
        with jit(self) as model:

            par = model.par
            sol = model.sol

            if par.do_print:
                print('Solving with G2EGM:')

            # a. solve retirement
            t0 = time.time()

            retirement.solve(sol,par)

            if par.do_print:
                print(f'solved retirement problem in {time.time()-t0:.2f} secs')

            # b. solve last period working
            t0 = time.time()

            last_period.solve(sol,par)

            if par.do_print:
                print(f'solved last period working in {time.time()-t0:.2f} secs')

            # c. solve working
            for t in reversed(range(par.T-1)):
                
                t0 = time.time()
                
                if par.do_print:
                    print(f' t = {t}:')
                
                # i. post decision
                t0_w = time.time()

                post_decision.compute(t,sol,par)

                par.time_w[t] = time.time()-t0_w
                if par.do_print:
                    print(f'   computed post decision value function in {par.time_w[t]:.2f} secs')

                # ii. EGM
                t0_EGM = time.time()
                
                G2EGM.solve(t,sol,par)
                
                par.time_egm[t] = time.time()-t0_EGM
                if par.do_print:
                    print(f'   applied G2EGM  in {par.time_egm[t]:.2f} secs')

                par.time_work[t] = time.time()-t0

            if par.do_print:
                print(f'solved working problem in {np.sum(par.time_work):.2f} secs')

    def solve_NEGM(self):
        """ solve with NEGM """
        
        with jit(self) as model:

            par = model.par
            sol = model.sol

            if par.do_print:
                print('Solving with NEGM:')

            # a. solve retirement
            t0 = time.time()

            retirement.solve(sol,par,G2EGM=False)

            if par.do_print:
                print(f'solved retirement problem in {time.time()-t0:.2f} secs')

            # b. solve last period working
            t0 = time.time()

            last_period.solve(sol,par,G2EGM=False)

            if par.do_print:
                print(f'solved last period working in {time.time()-t0:.2f} secs')

            # c. solve working  
            for t in reversed(range(par.T-1)):
                
                t0 = time.time()   
                
                if par.do_print:
                    print(f' t = {t}:')
                
                # i. post decision
                t0_w = time.time()

                post_decision.compute(t,sol,par,G2EGM=False)

                par.time_w[t] = time.time() - t0_w
                if par.do_print:
                    print(f'   computed post decision value function in {par.time_w[t]:.2f} secs')

                # ii. pure consumption problem
                t0_egm = time.time()
                
                NEGM.solve_pure_c(t,sol,par)
                
                par.time_egm[t] = time.time()-t0_egm
                if par.do_print:
                    print(f'   solved pure consumption problem in {par.time_egm[t]:.2f} secs')

                # iii. outer problem
                t0_vfi = time.time()
                
                NEGM.solve_outer(t,sol,par)
                
                par.time_vfi[t] = time.time()-t0_vfi
                if par.do_print:
                    print(f'   solved outer problem in {par.time_vfi[t] :.2f} secs')

                par.time_work[t] = time.time()-t0

            if par.do_print:
                print(f'solved working problem in {np.sum(par.time_work):.2f} secs')

    def checksums(self,Ts=None):
        """ calculate and print checksum """

        par = self.par
        sol = self.sol

        if Ts == 0:
            Ts = list(range(par.T))

        print('retirement')
        for t in Ts:
            print(f't = {t}, c: {np.sum(sol.c_ret[t,:]):.8f}')
            print(f't = {t}, v: {np.sum(sol.inv_v_ret[t,:]):.8f}')
            print('')
            
        print('working')
        for t in Ts:
            print('')
            print(f't = {t}, c: {np.sum(sol.c[t,:,:]):.8f}')   
            if par.solmethod == 'G2EGM': 
                print(f't = {t},  ucon: {np.nansum(sol.ucon_c[t,:,:]):.8f}')    
                print(f't = {t},  dcon: {np.nansum(sol.dcon_c[t,:,:]):.8f}')    
                print(f't = {t},  acon: {np.nansum(sol.acon_c[t,:,:]):.8f}')    
                print(f't = {t},  con: {np.nansum(sol.con_c[t,:,:]):.8f}')    
            print(f't = {t}, v: {np.sum(sol.inv_v[t,:,:]):.8f}')     
            if par.solmethod == 'G2EGM': 
                print(f't = {t},  ucon: {np.nansum(sol.ucon_c[t,:,:]):.8f}')    
                print(f't = {t},  dcon: {np.nansum(sol.dcon_c[t,:,:]):.8f}')    
                print(f't = {t},  acon: {np.nansum(sol.acon_c[t,:,:]):.8f}')    
                print(f't = {t},  con: {np.nansum(sol.con_c[t,:,:]):.8f}')               
            print(f't = {t}, vm: {np.sum(sol.inv_vm[t,:,:]):.8f}')     
            if par.solmethod == 'G2EGM': 
                print(f't = {t}, vn: {np.sum(sol.inv_vn[t,:,:]):.8f}')  
            if t < par.T-1:
                print(f't = {t}, w: {np.sum(sol.w[t,:,:]):.8f}')    
                print(f't = {t}, wa: {np.sum(sol.wa[t,:,:]):.8f}')    
                if par.solmethod == 'G2EGM': 
                    print(f't = {t}, wb: {np.sum(sol.wb[t,:,:]):.8f}')    
    
    def calculate_euler(self):
        """ calculate euler errors """
        
        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim

            simulate.euler(sim,sol,par)