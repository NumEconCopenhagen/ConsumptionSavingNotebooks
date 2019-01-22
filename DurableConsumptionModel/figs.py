import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set_style("whitegrid")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
import ipywidgets as widgets

from consav import linear_interp

def _decision_functions(model,t,ip,name):

    if name == 'discrete':
        _discrete(model,t,ip)
    elif name == 'adj':
        _adj(model,t,ip)
    elif name == 'keep':
        _keep(model,t,ip)
    elif name == 'post_decision' and t <= model.par.T-2:
        _w(model,t,ip)        

def decision_functions(model):
    widgets.interact(_decision_functions,
        model=widgets.fixed(model),
        t=widgets.Dropdown(description='t', 
            options=list(range(model.par.T)), value=0),
        ip=widgets.Dropdown(description='ip', 
            options=list(range(model.par.Np)), value=np.int(model.par.Np/2)),
        name=widgets.Dropdown(description='name', 
            options=['discrete','adj','keep','post_decision'], value='discrete')
        )

def _discrete(model,t,ip):

    par = model.par

    # a. interpolation
    db, m = np.meshgrid(par.grid_db,par.grid_m,indexing='ij')
    x = m + (1-par.tau)*db
    
    inv_v_adj = np.zeros(x.size)
    linear_interp.interp_1d_vec(par.grid_x,model.sol.inv_v_adj[t,ip,:,],x.ravel(),inv_v_adj)
    inv_v_adj = inv_v_adj.reshape(x.shape)

    # f. best discrete choice
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)

    I = inv_v_adj > model.sol.inv_v_keep[t,ip,:,:]

    x = m[I].ravel()
    y = db[I].ravel()
    ax.scatter(x,y,s=2,label='adjust')
    
    x = m[~I].ravel()
    y = db[~I].ravel()
    ax.scatter(x,y,s=2,label='keep')
        
    ax.set_title(f'optimal discrete choice ($t = {t}$, $p = {par.grid_p[ip]:.1f}$)',pad=10)

    legend = ax.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # g. details
    ax.grid(True)
    ax.set_xlabel('$m_t$')
    ax.set_xlim([par.grid_m[0],par.grid_m[-1]])
    ax.set_ylabel('$\\bar{d}_t$')
    ax.set_ylim([par.grid_db[0],par.grid_db[-1]])
    
    plt.show()

def _adj(model,t,ip):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(12,6))
    ax_b = fig.add_subplot(1,2,1)
    ax_v = fig.add_subplot(1,2,2)
    
    # c. plot consumption
    ax_b.plot(par.grid_x,sol.d_adj[t,ip,:],lw=2)
    ax_b.set_title(f'$d^{{adj}}$ ($t = {t}$, $p = {par.grid_p[ip]:.1f}$)',pad=10)

    # d. plot value function
    ax_v.plot(par.grid_x,sol.inv_v_adj[t,ip,:],lw=2)
    ax_v.set_title(f'neg. inverse $v^{{adj}}$ ($t = {t}$, $p = {par.grid_p[ip]:.1f}$)',pad=10)

    # e. details
    for ax in [ax_b,ax_v]:
        ax.grid(True)
        ax.set_xlabel('$x_t$')
        ax.set_xlim([par.grid_x[0],par.grid_x[-1]])

    plt.show()

def _w(model,t,ip):

    # a. unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1,projection='3d')

    db,a = np.meshgrid(par.grid_db, par.grid_a,indexing='ij')

    # c. plot consumption
    ax.plot_surface(db,a,sol.inv_w[t,ip,:,:],cmap=cm.viridis,edgecolor='none')
    ax.set_title(f'neg. inverse  $w$ ($t = {t}$, $p = {par.grid_p[ip]:.1f}$)',pad=10)

    # d. details
    ax.grid(True)
    ax.set_xlabel('$\\bar{d}_t$')
    ax.set_xlim([par.grid_db[0],par.grid_db[-1]])
    ax.set_ylabel('$a_t$')
    ax.set_ylim([par.grid_a[0],par.grid_a[-1]])
    ax.invert_xaxis()

    plt.show()

def _keep(model,t,ip):

    # unpack
    par = model.par
    sol = model.sol

    # b. figure
    fig = plt.figure(figsize=(12,6))
    ax_c = fig.add_subplot(1,2,1,projection='3d')
    ax_v = fig.add_subplot(1,2,2,projection='3d')

    db,m = np.meshgrid(par.grid_db, par.grid_m,indexing='ij')

    # c. plot consumption
    ax_c.plot_surface(db,m,sol.c_keep[t,ip,:,:],cmap=cm.viridis,edgecolor='none')
    ax_c.set_title(f'$c^{{keep}}$ ($t = {t}$, $p = {par.grid_p[ip]:.1f}$)',pad=10)

    # d. plot value function
    ax_v.plot_surface(db,m,sol.inv_v_keep[t,ip,:,:],cmap=cm.viridis,edgecolor='none')
    ax_v.set_title(f'neg. inverse $v^{{keep}}$ ($t = {t}$, $p = {par.grid_p[ip]:.1f}$)',pad=10)

    # e. details
    for ax in [ax_c,ax_v]:

        ax.grid(True)
        ax.set_xlabel('$\\bar{d}_t$')
        ax.set_xlim([par.grid_db[0],par.grid_db[-1]])
        ax.set_ylabel('$m_t$')
        ax.set_ylim([par.grid_m[0],par.grid_m[-1]])
        ax.invert_xaxis()

    plt.show()

def lifecycle(model):

    # a. unpack
    par = model.par
    sim = model.sim

    # b. figure
    fig = plt.figure(figsize=(12,12))

    simvarlist = [('p','$p_t$'),
                  ('db','$\\bar{d}_t$'),
                  ('m','$m_t$'),
                  ('c','$c_t$'),
                  ('a','$a_t$'),
                  ('discrete','adjuster share')]

    age = np.arange(par.T)
    for i,(simvar,simvarlatex) in enumerate(simvarlist):

        ax = fig.add_subplot(3,2,i+1)

        simdata = getattr(sim,simvar)[:par.T,:]

        ax.plot(age,np.mean(simdata,axis=1),lw=2)
        if simvar not in ['discrete']:
            ax.plot(age,np.percentile(simdata,25,axis=1),
                ls='--',lw=1,color='black')
            ax.plot(age,np.percentile(simdata,75,axis=1),
                ls='--',lw=1,color='black')
        ax.set_title(simvarlatex)
        ax.xaxis.set_ticks(age)

        ax.grid(True)
        if simvar in ['a','discrete']:
            ax.set_xlabel('age')

    plt.show()


def lifecycle_compare(model1,latex1,model2,latex2):

    # a. unpack
    par = model1.par
    sim1 = model1.sim
    sim2 = model2.sim

    # b. figure
    fig = plt.figure(figsize=(12,12))

    simvarlist = [('p','$p_t$'),
                  ('db','$\\bar{d}_t$'),
                  ('m','$m_t$'),
                  ('c','$c_t$'),
                  ('a','$a_t$'),
                  ('discrete','adjuster share')]

    age = np.arange(par.T)
    for i,(simvar,simvarlatex) in enumerate(simvarlist):

        ax = fig.add_subplot(3,2,i+1)

        simdata = getattr(sim1,simvar)[:par.T,:]
        ax.plot(age,np.mean(simdata,axis=1),lw=2,label=latex1)

        simdata = getattr(sim2,simvar)[:par.T,:]
        ax.plot(age,np.mean(simdata,axis=1),lw=2,label=latex2)

        ax.set_title(simvarlatex)
        ax.xaxis.set_ticks(age)

        ax.grid(True)
        if simvar in ['a','discrete']:
            ax.set_xlabel('age')
    
        ax.legend()
        
    plt.show()

