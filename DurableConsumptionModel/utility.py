from numba import njit

@njit(fastmath=True)
def func(c,d,par):
    return func_nopar(c,d,par.d_ubar,par.alpha,par.rho)

@njit(fastmath=True)
def func_nopar(c,d,d_ubar,alpha,rho):
    dtot = d+d_ubar
    c_total = c**alpha*dtot**(1.0-alpha)
    return c_total**(1-rho)/(1-rho)

@njit(fastmath=True)
def marg_func(c,d,par):
    return marg_func_nopar(c,d,par.d_ubar,par.alpha,par.rho)

@njit(fastmath=True)
def marg_func_nopar(c,d,d_ubar,alpha,rho):
    dtot = d+d_ubar
    c_power = alpha*(1.0-rho)-1.0
    d_power = (1.0-alpha)*(1.0-rho)
    return alpha*(c**c_power)*(dtot**d_power)

@njit(fastmath=True)
def inv_marg_func(q,d,par):
    dtot = d+par.d_ubar
    c_power = par.alpha*(1.0-par.rho)-1.0
    d_power = (1.0-par.alpha)*(1.0-par.rho)
    denom = par.alpha*dtot**d_power
    return (q/denom)**(1.0/c_power)