import time
import numpy as np
import numba as nb
import consav.cpptools as cpptools
import ctypes as ct

# a. test function
@nb.njit(parallel=True)
def example_par(X,Y,Z,NX,NY):

    # X is lenght NX
    # Y is lenght NY
    # Z is length NX

    for i in nb.prange(NX):
        Z[i] = 0
        for j in range(NY):
            Z[i] += np.exp(np.log(X[i]*Y[j]+0.001))/(X[i]*Y[j])-1

# b. settings
NX = 40000
NY = 40000

# c. random draws
np.random.seed(1998)
X = np.random.sample(NX)
Y = np.random.sample(NY)
Z = np.zeros(NX)

# d. compile cpp
funcs = [('fun',[ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),
                 ct.c_long,ct.c_long,ct.c_long])]
example_numba_vs = cpptools.link('example_numba_vs',funcs,use_openmp_with_vs=True,do_print=False)
#example_numba_intel = cpptools.link('example_numba_intel',funcs,do_print=False)

def wrapper_vs(X,Y,Z,NX,NY,threads):
    p_X = np.ctypeslib.as_ctypes(X)
    p_Y = np.ctypeslib.as_ctypes(Y)
    p_Z = np.ctypeslib.as_ctypes(Z)
    example_numba_vs.fun(p_X,p_Y,p_Z,NX,NY,threads)

def wrapper_intel(X,Y,Z,NX,NY,threads):
    p_X = np.ctypeslib.as_ctypes(X)
    p_Y = np.ctypeslib.as_ctypes(Y)
    p_Z = np.ctypeslib.as_ctypes(Z)
    example_numba_intel.fun(p_X,p_Y,p_Z,NX,NY,threads)

# e. test runs
NYtest = 2
Ytest = np.random.sample(NYtest)
example_par(X,Ytest,Z,NX,NYtest)

# f. timed runs
tic = time.time()
example_par(X,Y,Z,NX,NY)
toc = time.time()
print(f'  {"numba":10s} {np.sum(Z):.1f} in {toc-tic:.1f} secs')

if nb.config.THREADING_LAYER == 'omp':

    Z = np.zeros(NX)
    tic = time.time()
    wrapper_vs(X,Y,Z,NX,NY,nb.config.NUMBA_NUM_THREADS)
    toc = time.time()
    print(f'  {"C++, vs":10s} {np.sum(Z):.1f} in {toc-tic:.1f} secs')

    #Z = np.zeros(NX)
    #tic = time.time()
    #wrapper_intel(X,Y,Z,NX,NY,nb.config.NUMBA_NUM_THREADS)
    #toc = time.time()
    #print(f'  {"C++, intel":10s} {np.sum(Z):.1f} in {toc-tic:.1f} secs')