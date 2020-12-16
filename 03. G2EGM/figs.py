import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("seaborn-whitegrid")
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

color1 = np.array([3.0/255.0,103.0/255.0,166.0/255.0])
color2 = np.array([242.0/255.0,62.0/255.0,46.0/255.0])
color3 = np.array([3.0/255.0,166.0/255.0,166.0/255.0])
color4 = np.array([242.0/255.0,131.0/255.0,68.0/255.0])

def retirement(model):
    
    # a. unpack
    par = model.par
    sol = model.sol

    # b. settings
    fig_max_m = 5

    # c. figure
    for varname in ['c_ret','inv_v_ret']:
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        for t in range(par.T):
            I = sol.m_ret[t] < fig_max_m
            y = getattr(sol,varname)[t,I]
            ax.plot(sol.m_ret[t,I],y)

        # details
        ax.set_title(varname)
        ax.grid(True)
        ax.set_xlabel('$m_t$')
        ax.set_xlim([0,fig_max_m])

    plt.show()

def decision_functions(model,t):

    # a. settings
    fig_max_m = 5
    fig_max_n = 5
        
    # b. unpack
    par = model.par
    sol = model.sol
    
    # c. varnames
    for varname in ['c','d','inv_v']:
        
        if 'w' in varname:

            if t == par.T-1:
                continue
            else:
                x = par.grid_a_pd_nd.ravel()
                y = par.grid_b_pd_nd.ravel()
                I = (x < fig_max_m) & (y < fig_max_n)
            
        else:
            
            x = par.grid_m_nd.ravel()
            y = par.grid_n_nd.ravel()
            I = (x < fig_max_m) & (y < fig_max_n)
                
        # i. value
        value = getattr(sol,varname)[t].ravel()
        
        # ii. figure
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1,projection='3d')
        ax.scatter(x[I],y[I],value[I],s=1,c=value[I],cmap=cm.viridis)
        
        # iii. details
        ax.set_title(f't = {t}, {varname}')
        ax.grid(True)
        ax.set_xlabel('$m_t$')
        ax.set_xlim([0,fig_max_m])
        ax.set_ylabel('$n_t$')
        ax.set_xlim([0,fig_max_m])
    
    plt.show()

def segments(model,t):
    
    # a. unpack
    par = model.par
    sol = model.sol

    # b. settings
    fig_max_m = 5
    fig_max_n = 5

    # c. variables
    a = par.grid_m_nd - sol.c[t] - sol.d[t]
    d = np.fmax(sol.d[t],0)
    m = par.grid_m_nd
    n = par.grid_n_nd

    # d. indicators
    I = a < 1e-7
    a[I] = 0

    I = (m < fig_max_m) & (n < fig_max_n)
    Icon = (a == 0) & (d == 0) & (I == 1)
    Iucon = (a > 0) & (d > 0) & (I == 1)
    Iacon = (a == 0) & (d > 0) & (I == 1)
    Idcon = (a > 0) & (d == 0) & (I == 1)

    # e. figure
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)

    if Icon.sum() > 0: ax.scatter(m[Icon],n[Icon],s=4,color=color3,label='con')
    if Iacon.sum() > 0: ax.scatter(m[Iacon],n[Iacon],s=4,color=color1,label='acon')
    if Idcon.sum() > 0: ax.scatter(m[Idcon],n[Idcon],s=4,color=color2,label='dcon')
    if Iucon.sum() > 0: ax.scatter(m[Iucon],n[Iucon],s=4,color='black',label='ucon')

    # f. details
    ax.grid(True)
    legend = ax.legend(frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')    
    frame.set_alpha(1)    

    ax.set_xlabel('$m_t$')
    ax.set_xlim([0,fig_max_m])
    ax.set_ylabel('$n_t$')
    ax.set_xlim([0,fig_max_m])
        
    plt.show()