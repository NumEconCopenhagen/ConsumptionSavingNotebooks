import numpy as np
from numba import njit

@njit(fastmath=True)
def func(d,par):
    return par.chi*np.log(1+d)