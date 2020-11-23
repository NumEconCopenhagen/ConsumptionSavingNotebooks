//////////////////////////
// 1. external includes //
//////////////////////////

// standard C++ libraries
#include <windows.h>
#include <stdio.h>
#include <cmath>
#include <omp.h>

///////////////
// 2. macros //
///////////////

#define EXPORT extern "C" __declspec(dllexport)

/////////////
// 3. main //
/////////////

EXPORT void test_func(double *X, double *Y, double *Z, int NX, int NY, int threads){

    #pragma omp parallel num_threads(threads)
    {

    #pragma omp for   
    for(int i = 0; i < NX; i++){
        Z[i] = 0;
        for(int j = 0; j < NY; j++){
            Z[i] += exp(log(X[i]*Y[j]+0.001))/(X[i]*Y[j])-1;
        }
            
    }
    
    } // omp parallel

}

