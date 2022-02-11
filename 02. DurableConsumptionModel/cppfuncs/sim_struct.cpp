typedef struct sim_struct
{
 double* p0;
 double* d0;
 double* d10;
 double* d20;
 double* a0;
 double* utility;
 double* p;
 double* m;
 double* n;
 double* n1;
 double* n2;
 int* discrete;
 double* d;
 double* d1;
 double* d2;
 double* c;
 double* a;
 double* euler_error;
 double* euler_error_c;
 double* euler_error_rel;
 double* psi;
 double* xi;
} sim_struct;

double* get_double_p_sim_struct(sim_struct* x, char* name){

 if( strcmp(name,"p0") == 0 ){ return x->p0; }
 else if( strcmp(name,"d0") == 0 ){ return x->d0; }
 else if( strcmp(name,"d10") == 0 ){ return x->d10; }
 else if( strcmp(name,"d20") == 0 ){ return x->d20; }
 else if( strcmp(name,"a0") == 0 ){ return x->a0; }
 else if( strcmp(name,"utility") == 0 ){ return x->utility; }
 else if( strcmp(name,"p") == 0 ){ return x->p; }
 else if( strcmp(name,"m") == 0 ){ return x->m; }
 else if( strcmp(name,"n") == 0 ){ return x->n; }
 else if( strcmp(name,"n1") == 0 ){ return x->n1; }
 else if( strcmp(name,"n2") == 0 ){ return x->n2; }
 else if( strcmp(name,"d") == 0 ){ return x->d; }
 else if( strcmp(name,"d1") == 0 ){ return x->d1; }
 else if( strcmp(name,"d2") == 0 ){ return x->d2; }
 else if( strcmp(name,"c") == 0 ){ return x->c; }
 else if( strcmp(name,"a") == 0 ){ return x->a; }
 else if( strcmp(name,"euler_error") == 0 ){ return x->euler_error; }
 else if( strcmp(name,"euler_error_c") == 0 ){ return x->euler_error_c; }
 else if( strcmp(name,"euler_error_rel") == 0 ){ return x->euler_error_rel; }
 else if( strcmp(name,"psi") == 0 ){ return x->psi; }
 else if( strcmp(name,"xi") == 0 ){ return x->xi; }
 else {return NULL;}

}


int* get_int_p_sim_struct(sim_struct* x, char* name){

 if( strcmp(name,"discrete") == 0 ){ return x->discrete; }
 else {return NULL;}

}


