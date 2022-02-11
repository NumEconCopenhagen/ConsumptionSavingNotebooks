typedef struct par_struct
{
 bool do_2d;
 int T;
 double beta;
 double rho;
 double alpha;
 double d_ubar;
 double d1_ubar;
 double d2_ubar;
 double R;
 double tau;
 double tau1;
 double tau2;
 double delta;
 double delta1;
 double delta2;
 double gamma;
 double sigma_psi;
 int Npsi;
 double sigma_xi;
 int Nxi;
 double pi;
 double mu;
 int Np;
 double p_min;
 double p_max;
 int Nn;
 double n_max;
 int Nm;
 double m_max;
 int Nx;
 double x_max;
 int Na;
 double a_max;
 double sigma_p0;
 double mu_d0;
 double sigma_d0;
 double mu_a0;
 double sigma_a0;
 int simN;
 int sim_seed;
 double euler_cutoff;
 char* solmethod;
 int t;
 double tol;
 bool do_print;
 bool do_print_period;
 int cppthreads;
 bool do_simple_wq;
 bool do_marg_u;
 double* grid_p;
 double* grid_n;
 double* grid_m;
 double* grid_x;
 double* grid_a;
 double* psi;
 double* psi_w;
 double* xi;
 double* xi_w;
 int Nshocks;
 double* time_w;
 double* time_keep;
 double* time_adj;
 double* time_adj_full;
 double* time_adj_d1;
 double* time_adj_d2;
} par_struct;

bool get_bool_par_struct(par_struct* x, char* name){

 if( strcmp(name,"do_2d") == 0 ){ return x->do_2d; }
 else if( strcmp(name,"do_print") == 0 ){ return x->do_print; }
 else if( strcmp(name,"do_print_period") == 0 ){ return x->do_print_period; }
 else if( strcmp(name,"do_simple_wq") == 0 ){ return x->do_simple_wq; }
 else if( strcmp(name,"do_marg_u") == 0 ){ return x->do_marg_u; }
 else {return false;}

}


int get_int_par_struct(par_struct* x, char* name){

 if( strcmp(name,"T") == 0 ){ return x->T; }
 else if( strcmp(name,"Npsi") == 0 ){ return x->Npsi; }
 else if( strcmp(name,"Nxi") == 0 ){ return x->Nxi; }
 else if( strcmp(name,"Np") == 0 ){ return x->Np; }
 else if( strcmp(name,"Nn") == 0 ){ return x->Nn; }
 else if( strcmp(name,"Nm") == 0 ){ return x->Nm; }
 else if( strcmp(name,"Nx") == 0 ){ return x->Nx; }
 else if( strcmp(name,"Na") == 0 ){ return x->Na; }
 else if( strcmp(name,"simN") == 0 ){ return x->simN; }
 else if( strcmp(name,"sim_seed") == 0 ){ return x->sim_seed; }
 else if( strcmp(name,"t") == 0 ){ return x->t; }
 else if( strcmp(name,"cppthreads") == 0 ){ return x->cppthreads; }
 else if( strcmp(name,"Nshocks") == 0 ){ return x->Nshocks; }
 else {return -9999;}

}


double get_double_par_struct(par_struct* x, char* name){

 if( strcmp(name,"beta") == 0 ){ return x->beta; }
 else if( strcmp(name,"rho") == 0 ){ return x->rho; }
 else if( strcmp(name,"alpha") == 0 ){ return x->alpha; }
 else if( strcmp(name,"d_ubar") == 0 ){ return x->d_ubar; }
 else if( strcmp(name,"d1_ubar") == 0 ){ return x->d1_ubar; }
 else if( strcmp(name,"d2_ubar") == 0 ){ return x->d2_ubar; }
 else if( strcmp(name,"R") == 0 ){ return x->R; }
 else if( strcmp(name,"tau") == 0 ){ return x->tau; }
 else if( strcmp(name,"tau1") == 0 ){ return x->tau1; }
 else if( strcmp(name,"tau2") == 0 ){ return x->tau2; }
 else if( strcmp(name,"delta") == 0 ){ return x->delta; }
 else if( strcmp(name,"delta1") == 0 ){ return x->delta1; }
 else if( strcmp(name,"delta2") == 0 ){ return x->delta2; }
 else if( strcmp(name,"gamma") == 0 ){ return x->gamma; }
 else if( strcmp(name,"sigma_psi") == 0 ){ return x->sigma_psi; }
 else if( strcmp(name,"sigma_xi") == 0 ){ return x->sigma_xi; }
 else if( strcmp(name,"pi") == 0 ){ return x->pi; }
 else if( strcmp(name,"mu") == 0 ){ return x->mu; }
 else if( strcmp(name,"p_min") == 0 ){ return x->p_min; }
 else if( strcmp(name,"p_max") == 0 ){ return x->p_max; }
 else if( strcmp(name,"n_max") == 0 ){ return x->n_max; }
 else if( strcmp(name,"m_max") == 0 ){ return x->m_max; }
 else if( strcmp(name,"x_max") == 0 ){ return x->x_max; }
 else if( strcmp(name,"a_max") == 0 ){ return x->a_max; }
 else if( strcmp(name,"sigma_p0") == 0 ){ return x->sigma_p0; }
 else if( strcmp(name,"mu_d0") == 0 ){ return x->mu_d0; }
 else if( strcmp(name,"sigma_d0") == 0 ){ return x->sigma_d0; }
 else if( strcmp(name,"mu_a0") == 0 ){ return x->mu_a0; }
 else if( strcmp(name,"sigma_a0") == 0 ){ return x->sigma_a0; }
 else if( strcmp(name,"euler_cutoff") == 0 ){ return x->euler_cutoff; }
 else if( strcmp(name,"tol") == 0 ){ return x->tol; }
 else {return NAN;}

}


char* get_char_p_par_struct(par_struct* x, char* name){

 if( strcmp(name,"solmethod") == 0 ){ return x->solmethod; }
 else {return NULL;}

}


double* get_double_p_par_struct(par_struct* x, char* name){

 if( strcmp(name,"grid_p") == 0 ){ return x->grid_p; }
 else if( strcmp(name,"grid_n") == 0 ){ return x->grid_n; }
 else if( strcmp(name,"grid_m") == 0 ){ return x->grid_m; }
 else if( strcmp(name,"grid_x") == 0 ){ return x->grid_x; }
 else if( strcmp(name,"grid_a") == 0 ){ return x->grid_a; }
 else if( strcmp(name,"psi") == 0 ){ return x->psi; }
 else if( strcmp(name,"psi_w") == 0 ){ return x->psi_w; }
 else if( strcmp(name,"xi") == 0 ){ return x->xi; }
 else if( strcmp(name,"xi_w") == 0 ){ return x->xi_w; }
 else if( strcmp(name,"time_w") == 0 ){ return x->time_w; }
 else if( strcmp(name,"time_keep") == 0 ){ return x->time_keep; }
 else if( strcmp(name,"time_adj") == 0 ){ return x->time_adj; }
 else if( strcmp(name,"time_adj_full") == 0 ){ return x->time_adj_full; }
 else if( strcmp(name,"time_adj_d1") == 0 ){ return x->time_adj_d1; }
 else if( strcmp(name,"time_adj_d2") == 0 ){ return x->time_adj_d2; }
 else {return NULL;}

}


