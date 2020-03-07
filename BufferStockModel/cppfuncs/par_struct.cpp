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
 int Np;
 int Na;
 double tol;
 bool do_print;
 bool do_simple_w;
 int cppthreads;
 int simT;
 int simN;
 int sim_seed;
 double *grid_m;
 double *grid_p;
 double *grid_a;
 double *psi;
 double *psi_w;
 double *xi;
 double *xi_w;
 int Nshocks;
} par_struct;

