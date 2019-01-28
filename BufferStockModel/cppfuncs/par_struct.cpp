typedef struct par_struct
{
 int T;
 double beta;
 double rho;
 double R;
 double sigma_psi;
 int Npsi;
 double sigma_xi;
 int Nxi;
 double pi;
 double mu;
 int Nm;
 double *grid_m;
 int Np;
 double *grid_p;
 int Na;
 double *grid_a;
 int Nshocks;
 double *psi;
 double *psi_w;
 double *xi;
 double *xi_w;
 double tol;
 int simT;
 int simN;
 int sim_seed;
 bool do_print;
 bool do_simple_w;
 int cppthreads;
} par_struct;

