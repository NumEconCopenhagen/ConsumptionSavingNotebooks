#ifndef MAIN
#include "header.cpp"
#endif

EXPORT void compute_wq_negm(par_struct *par, sol_struct *sol)
{
    post_decision::compute_wq(par->t,sol,par,true);
}

EXPORT void solve_negm_keep(par_struct *par, sol_struct *sol)
{

    // unpack
    int t = par->t;
    
    int index_keep_t = index::d4(t,0,0,0,par->T,par->Np,par->Nn,par->Nm);
    auto inv_v = &sol->inv_v_keep[index_keep_t];
    auto inv_marg_u = &sol->inv_marg_u_keep[index_keep_t];
    auto c = &sol->c_keep[index_keep_t];

    int index_post_t = index::d4(t,0,0,0,par->T-1,par->Np,par->Nn,par->Na);
    auto inv_w = &sol->inv_w[index_post_t];
    auto q = &sol->q[index_post_t];
    auto q_c = &sol->q_c[index_post_t];
    auto q_m = &sol->q_m[index_post_t];

    #pragma omp parallel num_threads(par->cppthreads)
    {
        
        // temporary containers
        auto v_ast_vec = new double[par->Nm];

    #pragma omp for
    for(int i_p = 0; i_p < par->Np; i_p++){

        for(int i_n = 0; i_n < par->Nn; i_n++){
            
            // use euler equation
            double n = par->grid_n[i_n];
            for(int i_a = 0; i_a < par->Na; i_a++){
                int index_post = index::d3(i_p,i_n,i_a,par->Np,par->Nn,par->Na);
                q_c[index_post] = utility::inv_marg_func(q[index_post],n,par);
                q_m[index_post] = par->grid_a[i_a] + q_c[index_post];
            }
        
            // upperenvelope
            int index_keep = index::d3(i_p,i_n,0,par->Np,par->Nn,par->Nm);
            int index_post = index::d3(i_p,i_n,0,par->Np,par->Nn,par->Na);
            upperenvelope(par->grid_a,par->Na,&q_m[index_post],&q_c[index_post],&inv_w[index_post],true,
                          par->grid_m,par->Nm,
                          &c[index_keep],v_ast_vec,n,par);  

            // negative inverse
            for(int i_m = 0; i_m < par->Nm; i_m++){
                inv_v[index_keep+i_m] = -1.0/v_ast_vec[i_m];
                if(par->do_marg_u){
                    inv_marg_u[index_keep+i_m] = 1.0/utility::marg_func(c[index_keep+i_m],n,par);
                }
            }
        
        } // n
    
    } // p

        delete[] v_ast_vec;

    } // parallel

} // solve_keep