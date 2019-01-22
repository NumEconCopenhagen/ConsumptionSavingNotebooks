import numpy as np
from numba import njit

@njit(fastmath=True)
def p_plus_func(p,psi,par):
    p_plus = p*psi
    p_plus = np.fmax(p_plus,par.p_min) # lower bound
    p_plus = np.fmin(p_plus,par.p_max) # upper bound
    return p_plus

@njit(fastmath=True)
def db_plus_func(d,par):
    db_plus = (1-par.delta)*d
    db_plus = np.fmin(db_plus,par.db_max) # upper bound
    return db_plus

@njit(fastmath=True)
def m_plus_func(a,p_plus,xi_plus,par):
    y_plus = p_plus*xi_plus
    m_plus = par.R*a+ y_plus
    return m_plus

@njit(fastmath=True)
def x_plus_func(m_plus,db_plus,par):
    return m_plus + (1-par.tau)*db_plus