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

#include "par_struct.cpp"

EXPORT void setup_omp(){ // required when using vs
    SetEnvironmentVariable("OMP_WAIT_POLICY", "passive"); 
}

EXPORT void fun(par_struct *par){

    #pragma omp parallel num_threads(par->threads)
    {

    #pragma omp for   
    for(int i = 0; i < par->N; i++){
        par->Y[i] = par->X[i]*(par->a+par->b);
    }

    printf("omp_get_thread_num() = %2d, omp_get_num_procs() = %2d\n",omp_get_thread_num(),omp_get_num_procs());
    
    } // omp parallel

}

EXPORT void fun_nostruct(double *X, double *Y, int N, double a, double b, int threads){

    #pragma omp parallel num_threads(threads)
    {

    #pragma omp for      
    for(int i = 0; i < N; i++){
        Y[i] = X[i]*(a+b);
    }

    printf("omp_get_thread_num() = %2d, omp_get_num_procs() = %2d\n",omp_get_thread_num(),omp_get_num_procs());

    } // omp parallel
    
}