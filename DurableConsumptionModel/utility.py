from numba import njit

@njit(fastmath=True)
def func(c,d,par):
    dtot = d+par.db_ubar
    c_total = c**par.alpha*dtot**(1.0-par.alpha)
    return c_total**(1-par.rho)/(1-par.rho)

@njit(fastmath=True)
def marg_func(c,d,par):
    dtot = d+par.db_ubar
    c_power = par.alpha*(1.0-par.rho)-1.0
    d_factor = par.alpha*(dtot**((1.0-par.alpha)*(1.0-par.rho)))
    return (c**c_power)*d_factor

@njit(fastmath=True)
def inv_marg_func(q,d,par):
    dtot = d+par.db_ubar
    c_power = par.alpha*(1.0-par.rho)-1.0
    d_factor = par.alpha*(dtot**((1.0-par.alpha)*(1.0-par.rho)))
    return (q/d_factor)**(1.0/c_power)