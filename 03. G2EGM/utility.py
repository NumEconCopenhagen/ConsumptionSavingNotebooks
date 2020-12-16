from numba import njit

@njit(fastmath=True)
def func_ret(c,par):
    return c**(1-par.rho)/(1-par.rho)

@njit(fastmath=True)
def func(c,par):
    return c**(1-par.rho)/(1-par.rho) - par.alpha

@njit(fastmath=True)
def marg_func(c,par):
    return c**(-par.rho)

@njit(fastmath=True)
def inv_marg_func(q,par):
    return q**(-1.0/par.rho)