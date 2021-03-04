import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# from consav
from consav import linear_interp

def plot_value_function_convergence(model):

    par = model.par
    sol = model.sol

    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    for t in [par.T-1, par.T-2, par.T-6, par.T-11, 100, 50, 0]:
        if t > par.T-1 or t < 0: continue
        ax.plot(sol.m[t,:],-sol.inv_v[t,:],label=f'$n = {par.T-t}$')

    # limits
    ax.set_xlim([np.min(par.a_min), 5])
    ax.set_ylim([0, 1])

    # layout
    bbox = {'boxstyle':'square','ec':'white','fc':'white'}
    ax.text(1.5,0.4,f'$\\beta = {par.beta:.2f}$, $R = {par.R:.2f}$, $G = {par.G:.2f}$',bbox=bbox)
    ax.set_xlabel('$m_t$')
    ax.set_ylabel('$-1.0/v_t(m_t)$')
    ax.legend(loc='upper right',frameon=True)

    fig.savefig(f'figs/val_converge_{model.name}.pdf')

def plot_consumption_function_convergence(model):

    par = model.par
    sol = model.sol

    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    for t in [par.T-1, par.T-2, par.T-6, par.T-11, 100, 50, 0]:
        if t > par.T-1 or t < 0: continue
        ax.plot(sol.m[t,:],sol.c[t,:],label=f'$n = {par.T-t}$')

    # limits
    ax.set_xlim([np.min(par.a_min), 5])
    ax.set_ylim([0, 5])

    # layout
    bbox = {'boxstyle':'square','ec':'white','fc':'white'}
    ax.text(1.5,0.5,f'$\\beta = {par.beta:.2f}$, $R = {par.R:.2f}$, $G = {par.G:.2f}$',bbox=bbox)
    ax.set_xlabel('$m_t$')
    ax.set_ylabel(r'$c_t^{\star}(m_t)$')
    ax.legend(loc='upper left',frameon=True)

    fig.savefig(f'figs/cons_converge_{model.name}.pdf')

def plot_consumption_function_convergence_age(model):

    par = model.par
    sol = model.sol

    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    # consumption function for various ages
    for age in [25, 35, 45, 55, 65, 75, par.T+par.age_min-2, par.T+par.age_min-1]:
        ax.plot(sol.m[age-par.age_min],sol.c[age-par.age_min],label=f'age = {age}')

    # limits
    ax.set_xlim([min(par.a_min), 5])
    ax.set_ylim([0, 5])

    # layout
    bbox = {'boxstyle':'square','ec':'white','fc':'white'}
    ax.text(1.5,0.5,f'$\\beta = {par.beta:.2f}$, $R = {par.R:.2f}$, $G = {par.G:.2f}$',bbox=bbox)
    ax.set_xlabel('$m_t$')
    ax.set_ylabel('$c(m_t)$')
    ax.legend(loc='upper left',frameon=True)

    fig.savefig(f'figs/cons_converge_{model.name}.pdf')

def plot_consumption_function_pf(model):

    par = model.par
    sol = model.sol

    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    # perfect foresight consumption
    c_pf = (1-par.RI)*(sol.m[0,:]+(1-par.FHW)**(-1)-1)   

    # consumption function deviation from perfect foresight
    ax.plot(sol.m[0,:],sol.c[0,:]-c_pf,'-',lw=1.5)

    # limits
    ax.set_xlim([1, 500])
    ylim_now = ax.set_ylim()
    if np.max(np.abs(ylim_now)) < 1e-4:
        ax.set_ylim([-1,1])

    # layout
    ax.set_xlabel('$m_t$')
    ax.set_ylabel('$c(m_t) - c^{PF}(m_t)$')

    fig.savefig(f'figs/cons_converge_pf_{model.name}.pdf')

def plot_buffer_stock_target(model):

    par = model.par
    sol = model.sol

    # a. find a and avg. m_plus and c_plus
    
    # allocate
    a = np.nan*np.ones(par.Na+1)
    m_plus = np.nan*np.ones(par.Na+1)
    C_plus = np.nan*np.ones(par.Na+1)

    delta_log_C_plus = np.nan*np.ones(par.Na+1)
    delta_log_C_plus_approx_2 = np.nan*np.ones(par.Na+1)

    fac = 1.0/(par.G*par.psi_vec)
    for i_a in range(par.Na+1):

        # a. a and m
        a[i_a] = sol.m[0,i_a]-sol.c[0,i_a]            
        m_plus[i_a] = np.sum(par.w*(fac*par.R*a[i_a] + par.xi_vec))                

        # b. C_plus
        m_plus_vec = fac*par.R*a[i_a] + par.xi_vec            
        c_plus_vec = np.zeros(m_plus_vec.size)
        linear_interp.interp_1d_vec(sol.m[0,:],sol.c[0,:],m_plus_vec,c_plus_vec)
        C_plus_vec = par.G*par.psi_vec*c_plus_vec
        C_plus[i_a] = np.sum(par.w*C_plus_vec)

        # c. approx 
        if not (par.sigma_xi == 0 and par.sigma_psi == 0 and par.pi == 0) and sol.c[0,i_a] > 0:

            delta_log_C_plus[i_a] = np.sum(par.w*(np.log(par.G*C_plus_vec)))-np.log(sol.c[0,i_a])
            var_C_plus = np.sum(par.w*(np.log(par.G*C_plus_vec) - np.log(sol.c[0,i_a]) - delta_log_C_plus[i_a])**2)
            delta_log_C_plus_approx_2[i_a] = par.rho**(-1)*(np.log(par.R*par.beta)) + 2/par.rho*var_C_plus + np.log(par.G)

    # b. find target
    i = np.argmin(np.abs(m_plus-sol.m[0,:]))
    m_target = sol.m[0,i]

    # c. figure 1 - buffer-stock target
    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    # limits
    ax.set_xlim([np.min(par.a_min), 5])
    ax.set_ylim([0, 5])

    # layout
    bbox = {'boxstyle':'square','ec':'white','fc':'white'}
    ax.text(2.1,0.25,f'$\\beta = {par.beta:.2f}$, $R = {par.R:.2f}$, $G = {par.G:.2f}$',bbox=bbox)
    ax.set_xlabel('$m_t$')
    ax.set_ylabel('')

    # i. consumption
    ax.plot(sol.m[0,:],sol.c[0,:],'-',lw=1.5,label='$c(m_t)$')  
    ax.legend(loc='upper left',frameon=True)  
    fig.savefig(f'figs/buffer_stock_target_{model.name}_c.pdf')

    # ii. perfect foresight solution
    if par.FHW < 1 and par.RI < 1:

        c_pf = (1-par.RI)*(sol.m[0,:]+(1-par.FHW)**(-1)-1)   
        ax.plot(sol.m[0,:],c_pf,':',lw=1.5,color='black',label='$c^{PF}(m_t)$')

        ax.legend(loc='upper left',frameon=True)
        fig.savefig(f'figs/buffer_stock_target_{model.name}_pf.pdf')

    # iii. a    
    ax.plot(sol.m[0,:],a,'-',lw=1.5,label=r'$a_t=m_t-c^{\star}(m_t)$')
    ax.legend(loc='upper left',frameon=True)
    fig.savefig(f'figs/buffer_stock_target_{model.name}_a.pdf')

    # iv. m_plus
    ax.plot(sol.m[0,:],m_plus,'-',lw=1.5,label='$E[m_{t+1} | a_t]$')
    ax.legend(loc='upper left',frameon=True)
    fig.savefig(f'figs/buffer_stock_target_{model.name}_m_plus.pdf')
    
    # v. 45
    ax.plot([0,5],[0,5],'-',lw=1.5,color='black',label='45 degree')
    ax.legend(loc='upper left',frameon=True)
    fig.savefig(f'figs/buffer_stock_target_{model.name}_45.pdf')

    # vi. target            
    if not (par.sigma_xi == 0 and par.sigma_psi == 0 and par.pi == 0) == 'bs' and par.GI < 1:
        ax.plot([m_target,m_target],[0,5],'--',lw=1.5,color='black',label=f'target = {m_target:.2f}')

    ax.legend(loc='upper left',frameon=True)
    fig.savefig(f'figs/buffer_stock_target_{model.name}.pdf')

    # STOP
    if par.sigma_xi == 0 and par.sigma_psi == 0 and par.pi == 0:
        return

    # d. figure 2 - C ratio
    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    I = sol.c[0,:] > 0
    ax.plot(sol.m[0,I],(C_plus[I]/sol.c[0,I]),'-',lw=1.5,label='$E[C_{t+1}/C_t]$')
    ax.plot([m_target,m_target],[0,10],'--',lw=1.5,color='black',label='target')
    ax.plot([np.min(par.a_min),500],[par.G,par.G],':',lw=1.5,color='black',label='$G$')
    ax.plot([np.min(par.a_min),500],[(par.R*par.beta)**(1/par.rho),(par.R*par.beta)**(1/par.rho)],
        '-',lw=1.5,color='black',label=r'$(\beta R)^{1/\rho}$')

    # limit     
    ax.set_xlim([np.min(par.a_min),10])
    ax.set_ylim([0.95,1.1])

    # layout
    ax.set_xlabel('$m_t$')
    ax.set_ylabel('$C_{t+1}/C_t$')
    ax.legend(loc='upper right',frameon=True)

    fig.savefig(f'figs/cons_growth_{model.name}.pdf')

    # e. figure 3 - euler approx
    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    ax.plot(sol.m[0,:],delta_log_C_plus,'-',lw=1.5,label=r'$E[\Delta \log C_{t+1}]$')                

    ax.plot(sol.m[0,:],par.rho**(-1)*np.log(par.R*par.beta)*np.ones(par.Na+1)+np.log(par.G),'-',lw=1.5,label='1st order approx.')                
    ax.plot(sol.m[0,:],delta_log_C_plus_approx_2,'-',lw=1.5,label='2nd order approx.')                
    ax.plot([m_target,m_target],[-10 ,10],'--',lw=1.5,color='black',label='target')

    # limit     
    ax.set_xlim([np.min(par.a_min),10])
    ax.set_ylim([-0.03,0.12])

    # layout
    ax.set_xlabel('$m_t$')
    ax.set_ylabel(r'$E[\Delta \log C_{t+1}]$')
    ax.legend(loc='upper right',frameon=True)

    fig.savefig(f'figs/euler_approx_{model.name}.pdf')

####################
# simulation plots #
####################

def plot_simulate_cdf_cash_on_hand(model):

    par = model.par
    sim = model.sim

    # figure
    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    for t in [0,1,2,4,9,29,49,par.simT-1]:
        ecdf = ECDF(sim.m[:,t])
        ax.plot(ecdf.x,ecdf.y,lw=1.5,label=f'$t = {t}$')

    # limits
    ax.set_xlim([np.min(par.a_min),4])

    # layout  
    ax.set_xlabel('$m_t$')
    ax.set_ylabel('CDF')
    ax.legend(loc='upper right',frameon=True)

    fig.savefig(f'figs/sim_cdf_cash_on_hand_{model.name}.pdf')

def plot_simulate_consumption_growth(model):

    par = model.par
    sim = model.sim

    # 1. consumption growth
    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    y = np.mean(np.log(sim.C[:,1:])-np.log(sim.C[:,:-1]),axis=0)
    ax.plot(np.arange(par.simT-1),y,'-',lw=1.5,label=r'$E[\Delta\log(C_t)]$')

    y = np.log(np.mean(sim.C[:,1:],axis=0))-np.log(np.mean(sim.C[:,:-1],axis=0))
    ax.plot(np.arange(par.simT-1),y,'-',lw=1.5,
        label=r'$\Delta\log(E[C_t])$')
    
    ax.axhline(np.log(par.G),ls='-',lw=1.5,color='black',label='$\\log(G)$')
    ax.axhline(np.log(par.G)-0.5*par.sigma_psi**2,ls='--',lw=1.5,color='black',label=r'$\log(G)-0.5\sigma_{\psi}^2$')

    # layout  
    ax.set_xlabel('time')
    ax.set_ylabel('')
    ax.legend(loc='lower right',frameon=True)

    fig.savefig(f'figs/sim_cons_growth_{model.name}.pdf')

    # b. cash-on-hand
    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    ax.plot(np.arange(par.simT),np.mean(sim.m,axis=0),'-',lw=1.5,label='mean')
    ax.plot(np.arange(par.simT),np.percentile(sim.m,25,axis=0),'--',lw=1.5,color='black',label='25th percentile')
    ax.plot(np.arange(par.simT),np.percentile(sim.m,75,axis=0),'--',lw=1.5,color='black',label='75th percentile')

    # layout 
    ax.set_xlabel('time')
    ax.set_ylabel('$m_t$')
    ax.legend(loc='upper right',frameon=True)

    fig.savefig(f'figs/sim_cash_on_hand_{model.name}.pdf')

####################
# life-cycle plots #
####################

def plot_life_cycle_income(model):

    par = model.par
    sim = model.sim

    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    ax.plot(par.age_min+np.arange(1,par.simT),np.nanmean(sim.Y[:,1:],axis=0),'-',lw=1.5)

    # layout 
    ax.set_ylabel('income, $Y_t$')
    ax.set_xlabel('age')

    fig.savefig(f'figs/sim_Y_{model.name}.pdf')

def plot_life_cycle_cashonhand(model):

    par = model.par
    sim = model.sim

    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    ax.plot(par.age_min+np.arange(par.simT),np.mean(sim.M,axis=0),'-',lw=1.5)

    # layout 
    ax.set_ylabel('cash-on-hand, $M_t$')        
    ax.set_xlabel('age')

    fig.savefig(f'figs/sim_M_{model.name}.pdf')

def plot_life_cycle_consumption(model):

    par = model.par
    sim = model.sim

    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    ax.plot(par.age_min+np.arange(par.simT),np.mean(sim.C,axis=0),'-',lw=1.5)

    # layout 
    ax.set_ylabel('consumption, $C_t$')         
    ax.set_xlabel('age')

    fig.savefig(f'figs/sim_C_{model.name}.pdf')  

def plot_life_cycle_assets(model):

    par = model.par
    sim = model.sim

    fig = plt.figure(figsize=(6,4),dpi=100)
    ax = fig.add_subplot(1,1,1)

    ax.plot(par.age_min+np.arange(par.simT),np.mean(sim.A,axis=0),'-',lw=1.5)

    # layout 
    ax.set_ylabel('assets, $A_t$')         
    ax.set_xlabel('age')

    fig.savefig(f'figs/sim_A_{model.name}.pdf')
        
