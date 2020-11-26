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

#define MAX(X,Y) ((X)>(Y)?(X):(Y))
#define MIN(X,Y) ((X)<(Y)?(X):(Y))
#define BOUND(X,A,B) MIN(MAX(X,A),B)

#define EXPORT extern "C" __declspec(dllexport)

/////////////////
// 3. includes //
/////////////////

#include "HighResTimer_class.hpp"
#include "logs.cpp"

////////////////
// 4. structs //
////////////////

#include "par_struct.cpp"
#include "sol_struct.cpp"
#include "sim_struct.cpp"

/////////////
// 5. main //
/////////////

int binary_search(int imin, int Nx, double* x, double xi){
    int imid, half;

    // a. checks
    if(xi <= x[0]){
        return 0;
    } else if(xi >= x[Nx-2]) {
        return Nx-2;
    }

    // b. binary search
    while((half = Nx/2)){
        imid = imin + half;
        imin = (x[imid] <= xi) ? imid:imin;
        Nx  -= half;
    }

    return imin;

}

double interp_1d(double* grid, int Nx, double *value, double xi){

    // a. search
    int ix = binary_search(0,Nx,grid,xi);
    
    // b. relative positive
    double rel_x = (xi - grid[ix])/(grid[ix+1]-grid[ix]);
    
    // c. interpolate
    return value[ix] + rel_x * (value[ix+1]-value[ix]);

}

double interp_2d(double* grid1, double* grid2, int Nx1, int Nx2, double* value, double xi1, double xi2){

    // a. search in each dimension
    int ix1 = binary_search(0,Nx1,grid1,xi1);
    int ix2 = binary_search(0,Nx2,grid2,xi2);
    
    // b. relative positive
    double rel_x1 = (xi1 - grid1[ix1])/(grid1[ix1+1]-grid1[ix1]);
    double rel_x2 = (xi2 - grid2[ix2])/(grid2[ix2+1]-grid2[ix2]);
    
    // c. interpolate over inner dimension 
    double left = value[ix1*Nx2 + ix2] + rel_x2 * (value[ix1*Nx2 + ix2+1]-value[ix1*Nx2 + ix2]);
    double right = value[(ix1+1)*Nx2 + ix2] + rel_x2 * (value[(ix1+1)*Nx2 + ix2+1]-value[(ix1+1)*Nx2 + ix2]);

    // d. interpolate over outer dimension
    return left + rel_x1*(right-left);

}

void compute_wq(int t, sol_struct* sol, par_struct* par){

    // loop over outermost post-decision state
    #pragma omp for
    for(int ip = 0; ip < par->Np; ip++){

        for(int ia = 0; ia < par->Na; ia++){

            logs::write("log.txt",2," compute_wq: (ip,ia) = (%d,%d)\n",ip,ia);

            // initialize at zero
            int index = ip*par->Na+ia;
            sol->q[index] = 0.0;

            for(int ishock = 0; ishock < par->Nshocks; ishock++){
            
                // i. shocks
                double psi = par->psi[ishock];
                double psi_w = par->psi_w[ishock];
                double xi = par->xi[ishock];
                double xi_w = par->xi_w[ishock];

                // ii. next-period states;
                double p_plus = par->grid_p[ip]*psi;
                double y_plus = p_plus*xi;
                double m_plus = par->R*par->grid_a[ia] + y_plus;

                // iii. weights;
                double weight = psi_w*xi_w;

                // iv. interpolate and accumulate;
                double c_plus = interp_2d(par->grid_p,par->grid_m,par->Np,par->Nm,&(sol->c[(t+1)*par->Np*par->Nm]),p_plus,m_plus);
                sol->q[index] += weight*par->beta*par->R*pow(c_plus,-par->rho);

            } // shock
        } // a
    } // p

}

void solve_bellman_egm(int t, sol_struct* sol, par_struct* par, double* m_temp, double *c_temp){

    #pragma omp for
    for(int ip = 0; ip < par->Np; ip++){

        // a. invert Euler equation
        for(int ia = 0; ia < par->Na; ia++){
        
            logs::write("log.txt",2," solve_bellman_egm: (ip,ia) = %d,%d\n",ip,ia);
            
            c_temp[ia+1] = pow(sol->q[ip*par->Na+ia],-1.0/par->rho);
            m_temp[ia+1] = par->grid_a[ia] + c_temp[ia+1];
        
        } // a

        // b. re-interpolate consumption to common grid
        for(int im = 0; im < par->Nm; im++){
            
            logs::write("log.txt",2," solve_bellman_egm: (ip,im) = %d,%d\n",ip,im);

            int index = t*par->Np*par->Nm + ip*par->Nm + im;
            sol->c[index] = interp_1d(m_temp,par->Na+1,c_temp,par->grid_m[im]);
        
        } // m
       
    } // p

}

EXPORT void solve(par_struct* par, sol_struct* sol){

    logs::write("log.txt",0,"EGM.cpp (threads = %d)\n",par->cppthreads);

    #pragma omp parallel num_threads(par->cppthreads)
    {

        double *m_temp = new double[par->Na+1];
        double *c_temp = new double[par->Na+1];
        m_temp[0] = 0.0;
        c_temp[0] = 0.0;

        for(int t = par->T-1; t >= 0; t--){

            #pragma omp master
            logs::write("log.txt",1,"t = %d\n",t,par->T);

            // a. last period
            if(t == par->T-1){
                
                #pragma omp for
                for(int ip = 0; ip < par->Np; ip++){
                for(int im = 0; im < par->Nm; im++){
                    
                    logs::write("log.txt",2," last_period: (ip,im) = (%d,%d)\n",ip,im);

                    int index = t*par->Np*par->Nm + ip*par->Nm + im;
                    sol->c[index] = par->grid_m[im];

                } }

            } else {

                compute_wq(t,sol,par);
                #pragma omp barrier

                solve_bellman_egm(t,sol,par,m_temp,c_temp);
                #pragma omp barrier

            }

        }

        delete[] m_temp;
        delete[] c_temp;        

    }

    logs::write("log.txt",1,"done\n");

}