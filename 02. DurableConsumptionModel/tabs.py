import numpy as np

def euler_errors(models,postfix=''):

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
            
    keepers = []
    adjusters = []
    everybody = []
    for model in models:           
        
        keepers_now = model.sim.discrete[:-1,:].ravel() == 0
        adjusters_now = model.sim.discrete[:-1,:].ravel() > 0
        everybody_now = keepers_now | adjusters_now

        keepers.append(keepers_now)
        adjusters.append(adjusters_now)
        everybody.append(everybody_now)

    lines = []
    txt = 'All (average)'
    for i,model in enumerate(models):
        txt += f' & {avg_euler_error(model,everybody[i]):.3f}'
    txt += '\\\\ \n'
    lines.append(txt)

    txt = '\\,\\,5th percentile'
    for i,model in enumerate(models):
        txt += f' & {percentile_euler_error(model,everybody[i],5):.3f}'
    txt += '\\\\ \n'    
    lines.append(txt)

    txt = '\\,\\,95th percentile'
    for i,model in enumerate(models):
        txt += f' & {percentile_euler_error(model,everybody[i],95):.3f}'
    txt += '\\\\ \n'   
    lines.append(txt)

    txt = 'Adjusters (average)'
    for i,model in enumerate(models):
        txt += f' & {avg_euler_error(model,adjusters[i]):.3f}'
    txt += '\\\\ \n'    
    lines.append(txt)

    txt = 'Keepers (average)'
    for i,model in enumerate(models):
        txt += f' & {avg_euler_error(model,keepers[i]):.3f}'
    txt += '\\\\ \n'         
    lines.append(txt)

    with open(f'tabs_euler_errors{postfix}.tex', 'w') as txtfile:
        txtfile.writelines(lines)

def timings(models,speedup=False,postfix=''):

    lines = []

    txt = 'Total'
    for model in models:
        txt += f' & {np.sum(model.par.time_w+model.par.time_keep+model.par.time_adj)/60:.2f}'
    txt += '\\\\ \n'
    lines.append(txt)

    txt = 'Post-decision functions'
    for model in models:
        txt += f' & {np.sum(model.par.time_w)/60:.2f}'
    txt += '\\\\ \n'    
    lines.append(txt)

    txt = 'Keeper problem'
    for model in models:
        txt += f' & {np.sum(model.par.time_keep)/60:.2f}'
    txt += '\\\\ \n' 
    lines.append(txt)

    txt = 'Adjuster problem'
    for model in models:
        txt += f' & {np.sum(model.par.time_adj)/60:.2f}'
    txt += '\\\\ \n' 
    lines.append(txt)

    if speedup:
        basetime = np.sum(models[0].par.time_w+models[0].par.time_keep+models[0].par.time_adj)
        txt = 'Speed-up relative to VFI &'
        for model in models[1:]:
            txt += f' & {basetime/np.sum(model.par.time_w+model.par.time_keep+model.par.time_adj):.2f}'
        txt += '\\\\ \n' 
        lines.append(txt)

    with open(f'tabs_timings{postfix}.tex', 'w') as txtfile:
        txtfile.writelines(lines)

def simulation(models,postfix=''):

    lines = []

    txt = 'Expected discounted utility'
    for model in models:
        txt += f' & {np.mean(model.sim.utility):.3f}'
    txt += '\\\\ \n'
    lines.append(txt)

    if models[0].par.do_2d:
        txt = 'Adjuster share ($d^1_t \\neq n^1_t \\lor d^2_t \\neq n^2_t$)'
    else:
        txt = 'Adjuster share ($d_t \\neq n_t$)'

    for model in models:
        txt += f' & {np.mean(model.sim.discrete > 0):.3f}'
    txt += '\\\\ \n'
    lines.append(txt)
    
    txt = 'Average consumption ($c_t$)'
    for model in models:
        txt += f' & {np.mean(model.sim.c):.3f}'
    txt += '\\\\ \n'
    lines.append(txt)

    txt = 'Variance of consumption ($c_t$)'
    for model in models:
        txt += f' & {np.var(model.sim.c):.3f}'
    txt += '\\\\ \n'  
    lines.append(txt)

    if models[0].par.do_2d:
    
        txt = 'Average durable stock I ($d^1_t$)'
        for model in models:
            txt += f' & {np.mean(model.sim.d1):.3f}'
        txt += '\\\\ \n'
        lines.append(txt)

        txt = 'Variance of durable stock I ($d^1_t$)'
        for model in models:
            txt += f' & {np.var(model.sim.d1):.3f}'
        txt += '\\\\ \n'       
        lines.append(txt)

        txt = 'Average durable stock II ($d^2_t$)'
        for model in models:
            txt += f' & {np.mean(model.sim.d2):.3f}'
        txt += '\\\\ \n'
        lines.append(txt)

        txt = 'Variance of durable stock II ($d^2_t$)'
        for model in models:
            txt += f' & {np.var(model.sim.d2):.3f}'
        txt += '\\\\ \n'       
        lines.append(txt)

    else:

        txt = 'Average durable stock ($d_t$)'
        for model in models:
            txt += f' & {np.mean(model.sim.d):.3f}'
        txt += '\\\\ \n'
        lines.append(txt)

        txt = 'Variance of durable stock ($d_t$)'
        for model in models:
            txt += f' & {np.var(model.sim.d):.3f}'
        txt += '\\\\ \n'       
        lines.append(txt)

    with open(f'tabs_simulation{postfix}.tex', 'w') as txtfile:
        txtfile.writelines(lines)    

def all(models,speedup=False,postfix=''):

    euler_errors(models,postfix)        
    timings(models,speedup=speedup,postfix=postfix)        
    simulation(models,postfix)        