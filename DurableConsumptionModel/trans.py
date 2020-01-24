import numpy as np
from numba import njit

@njit(fastmath=True)
def p_plus_func(p,psi,par):
    p_plus = p*psi
    p_plus = np.fmax(p_plus,par.p_min) # lower bound
    p_plus = np.fmin(p_plus,par.p_max) # upper bound
    return p_plus

@njit(fastmath=True)
def n_plus_func(d,par):
    n_plus = (1-par.delta)*d
    n_plus = np.fmin(n_plus,par.n_max) # upper bound
    return n_plus

@njit(fastmath=True)
def n1_plus_func(d1,par):
    n1_plus = (1-par.delta1)*d1
    n1_plus = np.fmin(n1_plus,par.n_max) # upper bound
    return n1_plus

@njit(fastmath=True)
def n2_plus_func(d2,par):
    n2_plus = (1-par.delta2)*d2
    n2_plus = np.fmin(n2_plus,par.n_max) # upper bound
    return n2_plus

@njit(fastmath=True)
def m_plus_func(a,p_plus,xi_plus,par):
    y_plus = p_plus*xi_plus
    m_plus = par.R*a+ y_plus
    return m_plus

@njit(fastmath=True)
def x_plus_func(m_plus,n_plus,par):
    return m_plus + (1-par.tau)*n_plus