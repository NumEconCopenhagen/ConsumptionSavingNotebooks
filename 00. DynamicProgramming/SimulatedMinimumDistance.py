# -*- coding: utf-8 -*-
"""Simulated Minimum Distance

"""

##############
# 1. imports #
##############

import time
import numpy as np
from numba import njit, prange
from scipy import optimize

import matplotlib.pyplot as plt
import seaborn as sns 

plt.style.use('seaborn-whitegrid')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

markers = ['.','s','P','D','v','^','*']
style = ['-','--','-.',':','-','--','-.']

############
# 2. model #
############

class SimulatedMinimumDistanceClass():
    
    #########
    # setup #
    #########      

    def __init__(self,est_par,mom_func,datamoms=None,options=None):
        """ initialize """

        # a. the parameters that should be estimated
        self.est_par = est_par

        # b. the function that calculates moments 
        self.mom_func = mom_func

        # c. the moments in the data to be matched
        self.datamoms = datamoms

        # d. estimation options
        self.options = options

    def bootstrap_mom_var(self,data,num_obs,num_boot,num_moms,seed=9210):
        """ bootstrap moment covariance matrix """
        
        # a. set seed
        np.random.seed(seed)

        # b. allocate memory
        boot = np.empty((num_moms,num_boot))

        # c. draw random samples
        for b in range(num_boot):
            ids = np.random.randint(low=0,high=num_obs,size=num_obs)
            boot[:,b] = self.mom_func(data,ids)
        
        # d. calculate covariance
        Omega = np.cov(boot)

        # e. return Omega (scaled due to averages in moments)
        return Omega*num_obs

    def estimate(self,model,W,do_print_initial=True):
        """ estimate """

        # a. initial guesses
        est_par = self.est_par
        theta0 = np.array([val['guess'] for key,val in est_par.items()])
        names = [key for key,val in est_par.items()]

        # b. bounds
        lower = np.array([val['lower'] for key,val in est_par.items()])
        upper = np.array([val['upper'] for key,val in est_par.items()])

        # c. evaluate the objective function
        if do_print_initial: print(f'objective function at starting values: {self.obj_func(theta0,model,W,names,lower,upper)}')

        # d. call numerical solver
        method = 'nelder-mead' 
        res = optimize.minimize(self.obj_func,theta0,args=(model,W,names,lower,upper),
                                method=method)

        est = {'theta':res.x , 'obj_val':res.fun}
        for i,val in enumerate(res.x):
            key = names[i]
            est[key] = val

        return est

    def obj_func(self,theta,model,W,names,lower,upper):
        """ calculate objective function """

        # a. impose bounds and calculate penalty
        penalty = 0.0
        theta_clipped = theta.copy()
        for i in range(theta.size):
            
            # i. clip
            if (lower[i] != None) or (upper[i] != None):
                theta_clipped[i] = np.clip(theta_clipped[i],lower[i],upper[i])
            
            # ii. penalty
            penalty += 10_000.0*(theta[i]-theta_clipped[i])**2
        
        # b. calcualte the vector of differences between moments in data and in simulated data
        diff = self.diff_vec_func(theta_clipped,model,names)
        
        # c. return the objective function
        objval = diff.T @ W @ diff 
        return objval + penalty

    def diff_vec_func(self,theta,model,names):
        """ difference between data and simulated model moments """

        # a. update parameters in par
        for i in range(theta.size):
            setattr(model.par,names[i],theta[i])

        # b. solve model
        model.solve(do_print=False)
        
        # c. simulate model
        model.simulate(do_print=False)

        # calculate moments in simulated data
        moms_sim = self.mom_func(model.sim)
        
        # return the vector of differences
        return self.datamoms - moms_sim

    ###################
    # standard errors #
    ###################

    def num_grad(self,theta,model,names,step=1.0e-5,num_moms=None):
        """ calulcate numerical gradient vector """

        # a. determine number of moments and parameters
        num_par = theta.size
        if num_moms is None:
            num_moms = self.diff_vec_func(theta,model,names).size

        # b. allocate memory
        grad = np.empty((num_moms,num_par))

        # c. loop through parameters
        for par in range(num_par):

            # i. construct step as a function of scale of theta
            step_now = np.zeros(num_par)
            step_now[par] = np.fmax(np.abs(theta[par]*step),step)

            # ii. update theta's
            theta_plus = theta.copy() + step_now
            theta_minus = theta.copy() - step_now

            # iii. calcualte moments at these parameters
            mom_plus = self.diff_vec_func(theta_plus,model,names)
            mom_minus = self.diff_vec_func(theta_minus,model,names)

            # iv. store the gradient
            grad[:,par] = (mom_plus - mom_minus)/(2*step_now[par])

        # d. re-set all parameters
        for par in range(num_par):
            setattr(model.par,names[par],theta[par])

        return grad

    def calc_influence_function(self,theta,model,W):
        """ calculate influence function (Gamma) """

        # gradient wrt. theta parameters
        names = [key for key,val in self.est_par.items()]
        G = self.num_grad(theta,model,names)

        # return Gamma
        return - np.linalg.inv(G.T @ W @ G) @ G.T @ W , G
    
    ########################
    # sensitivity measures #
    ########################

    def informativeness_moments(self,grad,Omega,W):
        """ calculate informativeness of moments """
    
        info = dict()
        
        # a. calculate objects re-used below
        GW = grad.T @ W
        GWG = GW @ grad
        GWG_inv = np.linalg.inv(GWG)
        
        GSi = grad.T @ np.linalg.inv(Omega)
        GSiG = GSi @ grad
        
        Avar = GWG_inv @ (GW @ Omega @ GW.T) @ GWG_inv
        AvarOpt = np.linalg.inv(GSiG)
        
        # b. informativenss measures
        info['M1'] = - GWG_inv @ GW
        
        num_mom = len(Omega)
        num_par = len(grad[0])
        shape = (num_par,num_mom)
        info['M2'] = np.nan + np.zeros(shape)
        info['M3'] = np.nan + np.zeros(shape)
        info['M4'] = np.nan + np.zeros(shape)
        info['M5'] = np.nan + np.zeros(shape)
        info['M6'] = np.nan + np.zeros(shape)
        
        info['M2e'] = np.nan + np.zeros(shape)
        info['M3e'] = np.nan + np.zeros(shape)
        info['M4e'] = np.nan + np.zeros(shape)
        info['M5e'] = np.nan + np.zeros(shape)
        info['M6e'] = np.nan + np.zeros(shape)
        
        for k in range(num_mom):

            # pick out the kk'th element: Okk
            O = np.zeros((num_mom,num_mom))
            O[k,k] = 1
            
            M2kk = (np.linalg.inv(GSiG) @ (GSi @ O @ GSi.T)) @ np.linalg.inv(GSiG) # num_par-by-num_par
            M3kk = GWG_inv @ (GW @ O @ GW.T) @ GWG_inv
            M6kk =  - GWG_inv @ (grad.T@ O @ grad) @ Avar \
                    + GWG_inv @ (grad.T @ O @ Omega @ W @ grad) @ GWG_inv \
                    + GWG_inv @ (grad.T @ W @ Omega @ O @ grad) @ GWG_inv \
                    - Avar @ (grad.T @ O @ grad) @ GWG_inv # num_par-by-num_par
            
            info['M2'][:,k]  = np.diag(M2kk) # store only the diagonal: the effect on the variance of a given parameter from a slight change in the variance of the kth moment
            info['M3'][:,k]  = np.diag(M3kk) # store only the diagonal: the effect on the variance of a given parameter from a slight change in the variance of the kth moment
            info['M6'][:,k]  = np.diag(M6kk) # store only the diagonal: the effect on the variance of a given parameter from a slight change in the variance of the kth moment
            
            info['M2e'][:,k]  = info['M2'][:,k]/np.diag(AvarOpt) * Omega[k,k] # store only the diagonal: the effect on the variance of a given parameter from a slight change in the variance of the kth moment
            info['M3e'][:,k]  = info['M3'][:,k]/np.diag(Avar) * Omega[k,k] # store only the diagonal: the effect on the variance of a given parameter from a slight change in the variance of the kth moment
            info['M6e'][:,k]  = info['M6'][:,k]/np.diag(Avar) * W[k,k] # store only the diagonal: the effect on the variance of a given parameter from a slight change in the variance of the kth moment
            
            # remove the kth moment from the weight matrix and
            # calculate the asymptotic variance without this moment
            W_now = W.copy()
            W_now[k,:] = 0
            W_now[:,k] = 0
            
            GW_now = grad.T@W_now
            GWG_now = GW_now@grad
            Avar_now = (np.linalg.inv(GWG_now) @ (GW_now@ Omega @GW_now.T)) @ np.linalg.inv(GWG_now)
            
            info['M4'][:,k]  = np.diag(Avar_now) - np.diag(Avar)
            info['M4e'][:,k] = info['M4'][:,k] / np.diag(Avar)
            
            # optimal version
            Omega_now = np.delete(Omega,k,axis=0)
            Omega_now = np.delete(Omega_now,k,axis=1)
            grad_now = np.delete(grad,k,axis=0)
            AvarOpt_now = np.linalg.inv((grad_now.T @ np.linalg.inv(Omega_now)) @ grad_now)
            info['M5'][:,k]  = np.diag(AvarOpt_now) - np.diag(AvarOpt)
            info['M5e'][:,k] = info['M5'][:,k] / np.diag(AvarOpt)
        
        return info

    def sens_cali_brute_force(self,model,theta,W,cali_par_names,step=1.0e-5,do_print=True):
        """ brute force calculation of sensitivity to calibrated parameters """

        sens_brute = np.empty((theta.size,len(cali_par_names)))

        # a. estimate model for slightly changes calibrated paramaters
        for i,cali in enumerate(cali_par_names):

            # i. update calibrated parameters
            cali_val = getattr(model.par,cali) 
            setattr(model.par,cali,cali_val + step)

            # ii. print progress
            if do_print:
                print(f'now re-estimating with {cali} = {cali_val + step:2.6f} (original {cali_val:2.6f})')

            # iii. estimate model with these calibrated values
            est_now = self.estimate(model,W)
            theta_now = est_now['theta']

            # iv. calculate derivative
            sens_brute[:,i] = (theta_now - theta) / step

            # iv. re-set calibrated parameters
            setattr(model.par,cali,cali_val)

        # b. re-set all parameters in theta
        for i,(key,_) in enumerate(self.est_par.items()):
            setattr(model.par,key,theta[i])

        return sens_brute

    #########
    # plots #
    #########

    def plot(self,x_dict,y_dict,xlabel='',ylabel='',use_markers=False,hide_legend=False,save_path=None,fit=False):
        """ plot moments """

        # size settings
        fontsize = 17
        linewidth = 2

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        xmin = np.inf
        xmax = -np.inf
        n = 0
        ci = 0
        for name,_ in y_dict.items():

            if use_markers:
                marker = markers[n]
            else:
                marker = ''

            if fit and name.find('CI') >= 0:

                label_ci = '95% CI'
                if ci > 0: label_ci = ''
                ax.plot(x_dict[name],y_dict[name],marker='',linestyle=':',color='gray',label=label_ci,linewidth=linewidth)
                ci += 1 
            
            else:

                ax.plot(x_dict[name],y_dict[name],marker=marker,linestyle=style[n],color=colors[n],label=name,linewidth=linewidth)
            
            if min(x_dict[name]) < xmin: xmin = min(x_dict[name])
            if max(x_dict[name]) > xmax: xmax = max(x_dict[name])
            n += 1
            if n > 5: n = 0
        
        # set axis options
        plt.setp(ax.get_xticklabels(),fontsize=fontsize)
        plt.setp(ax.get_yticklabels(),fontsize=fontsize)

        ax.set_xlabel(xlabel,fontsize=fontsize)
        ax.set_ylabel(ylabel,fontsize=fontsize)

        ax.set_xlim([xmin,xmax])
        if not hide_legend: ax.legend(frameon=True,fontsize=fontsize)
        plt.tight_layout()

        if save_path is not None: fig.savefig(f'{save_path}.pdf')
        plt.show()

    def plot_heat(self,sens,est_par_names,cali_par_names,annot=True):
        """ plot heatmap """

        fs = 13
        cmap = sns.diverging_palette(220, 10, sep=10, n=100)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax = sns.heatmap(sens,annot=annot,fmt="2.2f",
                              annot_kws={"size": fs},
                              xticklabels=cali_par_names,
                              yticklabels=est_par_names,
                              center=0,
                              linewidth=.5,
                              cmap=cmap)
        
        plt.yticks(rotation=0) 
        ax.tick_params(axis='both', which='major', labelsize=fs)


        




    