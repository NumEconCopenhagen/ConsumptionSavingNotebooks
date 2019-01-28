from numba import njit

@njit(fastmath=True)
def func(c,d,par):
    dtot = d+par.d_ubar
    c_total = c**par.alpha*dtot**(1.0-par.alpha)
    return c_total**(1-par.rho)/(1-par.rho)

@njit(fastmath=True)
def marg_func(c,d,par):
    dtot = d+par.d_ubar
    c_power = par.alpha*(1.0-par.rho)-1.0
    d_power = (1.0-par.alpha)*(1.0-par.rho)
    return par.alpha*(c**c_power)*(dtot**d_power)

@njit(fastmath=True)
def inv_marg_func(q,d,par):
    dtot = d+par.d_ubar
    c_power = par.alpha*(1.0-par.rho)-1.0
    d_power = (1.0-par.alpha)*(1.0-par.rho)
    denom = par.alpha*dtot**d_power
    return (q/denom)**(1.0/c_power)