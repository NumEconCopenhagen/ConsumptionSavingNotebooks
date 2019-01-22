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

// a. generic
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

int binary_search(int imin, int Nx, double *x, double xi){
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

double interp_1d(double *grid1, int Nx1, double *value, double xi1){

    int j1 = binary_search(0,Nx1,grid1,xi1);
    
    double denom = grid1[j1+1]-grid1[j1];
    double nom = 0;
    for(int k1 = 0; k1 < 2; k1++){
        double nom_1 = k1 == 0 ? grid1[j1+1]-xi1 : xi1-grid1[j1];
        nom += nom_1*value[j1+k1];
    }
    return nom/denom;

}

double interp_2d(double *grid1, double *grid2, int Nx1, int Nx2, double *value, double xi1, double xi2){

    int j1 = binary_search(0,Nx1,grid1,xi1);
    int j2 = binary_search(0,Nx2,grid2,xi2);
    
    double denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2]);
    double nom = 0;
    for(int k1 = 0; k1 < 2; k1++){
        double nom_1 = k1 == 0 ? grid1[j1+1]-xi1 : xi1-grid1[j1];
        for(int k2 = 0; k2 < 2; k2++){
            double nom_2 = k2 == 0 ? grid2[j2+1]-xi2 : xi2-grid2[j2];  
            int index = (j1+k1)*Nx2 + j2+k2;     
            nom += nom_1*nom_2*value[index];
    } }
    return nom/denom;

}

double interp_3d(double *grid1, double *grid2, double *grid3, int Nx1, int Nx2, int Nx3, double *value, double xi1, double xi2, double xi3){

    int j1 = binary_search(0,Nx1,grid1,xi1);
    int j2 = binary_search(0,Nx2,grid2,xi2);
    int j3 = binary_search(0,Nx3,grid3,xi3);
    
    double nom_1_left = grid1[j1+1]-xi1;
    double nom_1_right = xi1-grid1[j1];

    double nom_2_left = grid2[j2+1]-xi2;
    double nom_2_right = xi2-grid2[j2];

    double nom_3_left = grid3[j3+1]-xi3;
    double nom_3_right = xi3-grid3[j3];

    double denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])*(grid3[j3+1]-grid3[j3]);
    double nom = 0;
    for(int k1 = 0; k1 < 2; k1++){
        double nom_1 = k1 == 0 ? nom_1_left : nom_1_right;
        for(int k2 = 0; k2 < 2; k2++){
            double nom_2 = k2 == 0 ? nom_2_left : nom_2_right;       
            for(int k3 = 0; k3 < 2; k3++){
                double nom_3 = k3 == 0 ? nom_3_left : nom_3_right;
                int index = (j1+k1)*Nx3*Nx2 + (j2+k2)*Nx3 + j3+k3;
                nom += nom_1*nom_2*nom_3*value[index];
    } } } 
    return nom/denom;

}

double utility_func(double c,double d, par_struct *par){
    double dtot = d+par->db_ubar;
    double c_total = pow(c,par->alpha)*pow(dtot,1.0-par->alpha);
    return pow(c_total,1.0-par->rho)/(1.0-par->rho);
}

double trans_p_plus_func(double p, double psi, par_struct *par){
    double p_plus = p*psi;
    p_plus = MAX(p_plus,par->p_min); 
    p_plus = MIN(p_plus,par->p_max); 
    return p_plus;
}

double trans_db_plus_func(double d,par_struct *par){
    double db_plus = (1.0-par->delta)*d;
    db_plus = MIN(db_plus,par->db_max);
    return db_plus;
}

double trans_m_plus_func(double a, double p_plus, double xi_plus, par_struct *par){
    double y_plus = p_plus*xi_plus;
    double m_plus = par->R*a+ y_plus;
    return m_plus;

}

double trans_x_plus_func(double m_plus, double db_plus,par_struct *par){
    return m_plus + (1.0-par->tau)*db_plus;
}

double obj(double c,double p, double db, double m, double* inv_v_plus_keep, double* inv_v_plus_adj, par_struct *par){

    // a. end-of-period assets
    double a = m-c;
    
    // b. continuation value
    double w = 0;
    for(int ishock = 0; ishock < par->Nshocks; ishock++){
            
        // i. shocks
        double psi = par->psi[ishock];
        double psi_w = par->psi_w[ishock];
        double xi = par->xi[ishock];
        double xi_w = par->xi_w[ishock];

        // ii. next-period states
        double p_plus = trans_p_plus_func(p,psi,par);
        double db_plus = trans_db_plus_func(db,par);
        double m_plus = trans_m_plus_func(a,p_plus,xi,par);
        double x_plus = trans_x_plus_func(m_plus,db_plus,par);
                
        // iii. weight
        double weight = psi_w*xi_w;
        
        // iv. update
        double inv_v_plus_keep_now = interp_3d(par->grid_p,par->grid_db,par->grid_m,par->Np,par->Ndb,par->Nm,inv_v_plus_keep,p_plus,db_plus,m_plus);
        double inv_v_plus_adj_now = interp_2d(par->grid_p,par->grid_x,par->Np,par->Nx,inv_v_plus_adj,p_plus,x_plus);
        
        double v_plus_now = -HUGE_VAL;
        if(inv_v_plus_keep_now > inv_v_plus_adj_now && inv_v_plus_keep_now > 0){
            v_plus_now = -1.0/inv_v_plus_keep_now;
        } else if(inv_v_plus_adj_now > 0){
            v_plus_now = -1.0/inv_v_plus_adj_now;
        }
        w += weight*par->beta*v_plus_now;

    }

    // c. total value
    double value_of_choice = utility_func(c,db,par) + w;

    return value_of_choice;

}

EXPORT void solve_keep(par_struct *par, sol_struct *sol, sim_struct *sim){
    
    logs::write("log.txt",0,"");

    // unpack
    double *inv_v = &sol->inv_v_keep[par->t*par->Np*par->Ndb*par->Nm];
    double *c = &sol->c_keep[par->t*par->Np*par->Ndb*par->Nm];

    // keep: loop over outer states
    #pragma omp parallel num_threads(par->cppthreads)
    {

    #pragma omp for
    for(int ip = 0; ip < par->Np; ip++){

    for(int idb = 0; idb < par->Ndb; idb++){
            
        // outer states
        double p = par->grid_p[ip];
        double db = par->grid_db[idb];

        for(int im = 0; im < par->Nm; im++){
            
            int index = ip*par->Ndb*par->Nm + idb*par->Nm + im;

            if(im == 0){
                c[index] = 0;
                inv_v[index] = 0;
                continue;
            }

            // a. cash-on-hand
            double m = par->grid_m[im];
                
            // b. optimal choice
            double c_low = MIN(m/2,1e-8);
            double c_high = m;

            double v_max = -HUGE_VAL;
            double c_max = 0;
            
            for(int ic = 0; ic < par->Nc_keep; ic++){

                double c_now = c_low + par->grid_c_keep[ic]*(c_high-c_low);
                double v_now = obj( c_now,p,db,m,
                                    &sol->inv_v_keep[(par->t+1)*par->Np*par->Ndb*par->Nm],
                                    &sol->inv_v_adj[(par->t+1)*par->Np*par->Nx],par);

                if(v_now > v_max){
                    v_max = v_now;
                    c_max = c_now;
                }
            
            } // c

            // c. optimal value
            c[index] = c_max;
            inv_v[index] = -1.0/v_max;

        } // m
    
    } } // p and db

    } // parallel

}

EXPORT void solve_adj(par_struct *par, sol_struct *sol, sim_struct *sim){

    // unpack
    double* inv_v = &sol->inv_v_adj[par->t*par->Np*par->Nx];
    double* d = &sol->d_adj[par->t*par->Np*par->Nx];
    double* c = &sol->c_adj[par->t*par->Np*par->Nx];

    // keep: loop over outer states
    #pragma omp parallel num_threads(par->cppthreads)
    {

    #pragma omp for
    for(int ip = 0; ip < par->Np; ip++){
            
            // outer states
            double p = par->grid_p[ip];

            // loop over x state
            for(int ix = 0; ix < par->Nx; ix++){
                
                int index = ip*par->Nx+ix;

                if(ix == 0){
                    c[index] = 0;
                    d[index] = 0;
                    inv_v[index] = 0;
                    continue;
                }

                // a. cash-on-hand
                double x = par->grid_x[ix];

                double d_low = 0;
                double d_high = x;

                double v_max = -HUGE_VAL;
                double d_max = 0;       
                double c_max = 0;               
                
                // loop over d choice
                 for(int id = 0; id < par->Nd_adj; id++){
                    
                    double d_now = d_low + par->grid_d_adj[id]*(d_high-d_low);
                    double m = x-d_now;

                    double c_low = MIN(x/2,1e-8);
                    double c_high = x;

                    for(int ic = 0; ic < par->Nc_adj; ic++){

                        double c_now = c_low + par->grid_c_adj[ic]*(c_high-c_low);
                        double v_now = obj( c_now,p,d_now,m,
                                            &sol->inv_v_keep[(par->t+1)*par->Np*par->Ndb*par->Nm],
                                            &sol->inv_v_adj[(par->t+1)*par->Np*par->Nx],par);

                        if(v_now > v_max){
                            v_max = v_now;
                            d_max = d_now;
                            c_max = c_now;
                        }
                    
                    } // c

                } // d

                // d. optimal value
                d[index] = d_max;
                c[index] = c_max;
                inv_v[index] = -1/v_max;
            
            } // x
        } // p

    } // parallel

}

// required when using visual studio
EXPORT void setup_omp(){ SetEnvironmentVariable("OMP_WAIT_POLICY", "passive"); }