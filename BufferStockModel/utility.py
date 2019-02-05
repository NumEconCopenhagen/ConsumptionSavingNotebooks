from numba import njit

@njit
def func(c,par):
    return c**(1-par.rho)/(1-par.rho)

@njit
def marg_func(c,par):
    return c**(-par.rho)

@njit
def inv_marg_func(q,par):
    return q**(-1/par.rho)