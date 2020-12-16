from numba import njit

@njit(fastmath=True)
def func(c,d,par):
    return func_nopar(c,d,par.d_ubar,par.alpha,par.rho)

@njit(fastmath=True)
def func_2d(c,d1,d2,par):
    return func_nopar_2d(c,d1,d2,par.d1_ubar,par.d2_ubar,par.alpha,par.rho,par.gamma)

@njit(fastmath=True)
def func_nopar_2d(c,d1,d2,d1_ubar,d2_ubar,alpha,rho,gamma):
    d1tot = d1+d1_ubar
    d2tot = d2+d2_ubar
    c_total = c**alpha*d1tot**(gamma*(1.0-alpha))*d2tot**((1-gamma)*(1.0-alpha))
    return c_total**(1-rho)/(1-rho)

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
def marg_func_2d(c,d1,d2,par):
    return marg_func_nopar_2d(c,d1,d2,par.d1_ubar,par.d2_ubar,par.alpha,par.rho,par.gamma)
    
@njit(fastmath=True)
def marg_func_nopar_2d(c,d1,d2,d1_ubar,d2_ubar,alpha,rho,gamma):
    d1tot = d1+d1_ubar
    d2tot = d2+d2_ubar
    c_power = alpha*(1.0-rho)-1.0
    d1_power = gamma*(1.0-alpha)*(1.0-rho)
    d2_power = (1.0-gamma)*(1.0-alpha)*(1.0-rho)
    return alpha*(c**c_power)*(d1tot**d1_power)*(d2tot**d2_power)

@njit(fastmath=True)
def inv_marg_func(q,d,par):
    dtot = d+par.d_ubar
    c_power = par.alpha*(1.0-par.rho)-1.0
    d_power = (1.0-par.alpha)*(1.0-par.rho)
    denom = par.alpha*dtot**d_power
    return (q/denom)**(1.0/c_power)

@njit(fastmath=True)
def inv_marg_func_2d(q,d1,d2,par):
    d1tot = d1+par.d1_ubar
    d2tot = d2+par.d2_ubar
    c_power = par.alpha*(1.0-par.rho)-1.0
    d1_power = par.gamma*(1.0-par.alpha)*(1.0-par.rho)
    d2_power = (1-par.gamma)*(1.0-par.alpha)*(1.0-par.rho)
    denom = par.alpha*(d1tot**d1_power)*(d2tot**d2_power)
    return (q/denom)**(1.0/c_power)    