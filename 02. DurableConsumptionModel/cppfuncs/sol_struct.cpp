typedef struct sol_struct
{
 double* c_keep;
 double* inv_v_keep;
 double* inv_marg_u_keep;
 double* d_adj;
 double* c_adj;
 double* inv_v_adj;
 double* inv_marg_u_adj;
 double* inv_w;
 double* q;
 double* q_c;
 double* q_m;
 double* c_keep_2d;
 double* inv_v_keep_2d;
 double* inv_marg_u_keep_2d;
 double* d1_adj_full_2d;
 double* d2_adj_full_2d;
 double* c_adj_full_2d;
 double* inv_v_adj_full_2d;
 double* inv_marg_u_adj_full_2d;
 double* d1_adj_d1_2d;
 double* c_adj_d1_2d;
 double* inv_v_adj_d1_2d;
 double* inv_marg_u_adj_d1_2d;
 double* d2_adj_d2_2d;
 double* c_adj_d2_2d;
 double* inv_v_adj_d2_2d;
 double* inv_marg_u_adj_d2_2d;
 double* inv_w_2d;
 double* q_2d;
 double* q_c_2d;
 double* q_m_2d;
} sol_struct;

double* get_double_p_sol_struct(sol_struct* x, char* name){

 if( strcmp(name,"c_keep") == 0 ){ return x->c_keep; }
 else if( strcmp(name,"inv_v_keep") == 0 ){ return x->inv_v_keep; }
 else if( strcmp(name,"inv_marg_u_keep") == 0 ){ return x->inv_marg_u_keep; }
 else if( strcmp(name,"d_adj") == 0 ){ return x->d_adj; }
 else if( strcmp(name,"c_adj") == 0 ){ return x->c_adj; }
 else if( strcmp(name,"inv_v_adj") == 0 ){ return x->inv_v_adj; }
 else if( strcmp(name,"inv_marg_u_adj") == 0 ){ return x->inv_marg_u_adj; }
 else if( strcmp(name,"inv_w") == 0 ){ return x->inv_w; }
 else if( strcmp(name,"q") == 0 ){ return x->q; }
 else if( strcmp(name,"q_c") == 0 ){ return x->q_c; }
 else if( strcmp(name,"q_m") == 0 ){ return x->q_m; }
 else if( strcmp(name,"c_keep_2d") == 0 ){ return x->c_keep_2d; }
 else if( strcmp(name,"inv_v_keep_2d") == 0 ){ return x->inv_v_keep_2d; }
 else if( strcmp(name,"inv_marg_u_keep_2d") == 0 ){ return x->inv_marg_u_keep_2d; }
 else if( strcmp(name,"d1_adj_full_2d") == 0 ){ return x->d1_adj_full_2d; }
 else if( strcmp(name,"d2_adj_full_2d") == 0 ){ return x->d2_adj_full_2d; }
 else if( strcmp(name,"c_adj_full_2d") == 0 ){ return x->c_adj_full_2d; }
 else if( strcmp(name,"inv_v_adj_full_2d") == 0 ){ return x->inv_v_adj_full_2d; }
 else if( strcmp(name,"inv_marg_u_adj_full_2d") == 0 ){ return x->inv_marg_u_adj_full_2d; }
 else if( strcmp(name,"d1_adj_d1_2d") == 0 ){ return x->d1_adj_d1_2d; }
 else if( strcmp(name,"c_adj_d1_2d") == 0 ){ return x->c_adj_d1_2d; }
 else if( strcmp(name,"inv_v_adj_d1_2d") == 0 ){ return x->inv_v_adj_d1_2d; }
 else if( strcmp(name,"inv_marg_u_adj_d1_2d") == 0 ){ return x->inv_marg_u_adj_d1_2d; }
 else if( strcmp(name,"d2_adj_d2_2d") == 0 ){ return x->d2_adj_d2_2d; }
 else if( strcmp(name,"c_adj_d2_2d") == 0 ){ return x->c_adj_d2_2d; }
 else if( strcmp(name,"inv_v_adj_d2_2d") == 0 ){ return x->inv_v_adj_d2_2d; }
 else if( strcmp(name,"inv_marg_u_adj_d2_2d") == 0 ){ return x->inv_marg_u_adj_d2_2d; }
 else if( strcmp(name,"inv_w_2d") == 0 ){ return x->inv_w_2d; }
 else if( strcmp(name,"q_2d") == 0 ){ return x->q_2d; }
 else if( strcmp(name,"q_c_2d") == 0 ){ return x->q_c_2d; }
 else if( strcmp(name,"q_m_2d") == 0 ){ return x->q_m_2d; }
 else {return NULL;}

}


