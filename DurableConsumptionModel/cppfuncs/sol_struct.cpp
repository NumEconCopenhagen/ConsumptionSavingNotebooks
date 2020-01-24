typedef struct sol_struct
{
 double *c_keep;
 double *inv_v_keep;
 double *inv_marg_u_keep;
 double *inv_v_adj;
 double *inv_marg_u_adj;
 double *c_adj;
 double *d_adj;
 double *inv_w;
 double *q;
 double *q_c;
 double *q_m;
 double *c_keep_2d;
 double *inv_v_keep_2d;
 double *inv_marg_u_keep_2d;
 double *inv_v_adj_full_2d;
 double *inv_marg_u_adj_full_2d;
 double *c_adj_full_2d;
 double *d1_adj_full_2d;
 double *d2_adj_full_2d;
 double *inv_v_adj_d1_2d;
 double *inv_marg_u_adj_d1_2d;
 double *c_adj_d1_2d;
 double *d1_adj_d1_2d;
 double *inv_v_adj_d2_2d;
 double *inv_marg_u_adj_d2_2d;
 double *c_adj_d2_2d;
 double *d2_adj_d2_2d;
 double *inv_w_2d;
 double *q_2d;
 double *q_c_2d;
 double *q_m_2d;
} sol_struct;

