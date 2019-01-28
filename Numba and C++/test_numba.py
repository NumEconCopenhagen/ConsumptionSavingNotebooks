import time
import numpy as np
import numba as nb
import consav.cpptools as cpptools
import ctypes as ct

# a. test function
@nb.njit(parallel=True)
def test(X,Y,Z,NX,NY):

    # X is lenght NX
    # Y is lenght NY
    # Z is length NX

    for i in nb.prange(NX):
        for j in range(NY):
            Z[i] += np.exp(np.log(X[i]*Y[j]))/(X[i]*Y[j])-1

@nb.njit(parallel=True,fastmath=True)
def test_fast(X,Y,Z,NX,NY):
    for i in nb.prange(NX):
        for j in range(NY):
            Z[i] += np.exp(np.log(X[i]*Y[j]))/(X[i]*Y[j])-1

# b. settings
NX = 20000
NY = 20000

# c. random draws
np.random.seed(1998)
X = np.random.sample(NX)
Y = np.random.sample(NY)
Z = np.zeros(NX)

# d. compile cpp
funcs = [('fun',[ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),
                 ct.c_long,ct.c_long,ct.c_long])]
test_numba_vs = cpptools.link('test_numba_vs',funcs,use_openmp_with_vs=True,do_print=False)
test_numba_intel = cpptools.link('test_numba_intel',funcs,do_print=False)

def wrapper_vs(X,Y,Z,NX,NY,threads):
    p_X = np.ctypeslib.as_ctypes(X)
    p_Y = np.ctypeslib.as_ctypes(Y)
    p_Z = np.ctypeslib.as_ctypes(Z)
    test_numba_vs.fun(p_X,p_Y,p_Z,NX,NY,threads)

def wrapper_intel(X,Y,Z,NX,NY,threads):
    p_X = np.ctypeslib.as_ctypes(X)
    p_Y = np.ctypeslib.as_ctypes(Y)
    p_Z = np.ctypeslib.as_ctypes(Z)
    test_numba_intel.fun(p_X,p_Y,p_Z,NX,NY,threads)

# e. test runs
NYtest = 2
Ytest = np.random.sample(NYtest)
test(X,Ytest,Z,NX,NYtest)
test_fast(X,Ytest,Z,NX,NYtest)

# f. timed runs
tic = time.time()
test(X,Y,Z,NX,NY)
toc = time.time()
print(f'  test {np.sum(Z):.8f} in {toc-tic:.1f} secs')

Z = np.zeros(NX)
tic = time.time()
test_fast(X,Y,Z,NX,NY)
toc = time.time()
print(f'  test (fastmath=true) {np.sum(Z):.8f} in {toc-tic:.1f} secs')

if nb.config.THREADING_LAYER == 'omp':

    Z = np.zeros(NX)
    tic = time.time()
    wrapper_vs(X,Y,Z,NX,NY,nb.config.NUMBA_NUM_THREADS)
    toc = time.time()
    print(f'  test (cpp, vs) {np.sum(Z):.8f} in {toc-tic:.1f} secs')

    Z = np.zeros(NX)
    tic = time.time()
    wrapper_intel(X,Y,Z,NX,NY,nb.config.NUMBA_NUM_THREADS)
    toc = time.time()
    print(f'  test (cpp, intel) {np.sum(Z):.8f} in {toc-tic:.1f} secs')