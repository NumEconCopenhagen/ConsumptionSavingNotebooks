typedef struct sim_struct
{
 double *utility;
 double *p0;
 double *d0;
 double *a0;
 double *p;
 double *n;
 double *m;
 double *x;
 double *c;
 double *d;
 double *a;
 double *psi;
 double *xi;
 int *discrete;
 double *euler_error;
 double *euler_error_c;
 double *euler_error_rel;
} sim_struct;

