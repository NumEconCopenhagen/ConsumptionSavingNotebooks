#ifndef MAIN
#define POST_DECISION
#include "header.cpp"
#endif

namespace post_decision {

void compute_wq(int t, sol_struct *sol, par_struct *par, bool compute_q)
{

    // unpack
    auto inv_w = &sol->inv_w[index::d4(t,0,0,0,par->T-1,par->Np,par->Nn,par->Na)];
    auto q = &sol->q[index::d4(t,0,0,0,par->T-1,par->Np,par->Nn,par->Na)];

    // loop over outermost post-decision state
    #pragma omp parallel num_threads(par->cppthreads)
    {
    
        // allocate temporary containers
        auto m_plus = new double[par->Na]; // container, same lenght as grid_a
        auto x_plus = new double[par->Na];
        auto w = new double[par->Na]; 
        auto inv_v_keep_plus = new double[par->Na];
        auto inv_marg_u_keep_plus = new double[par->Na];
        auto inv_v_adj_plus = new double[par->Na];
        auto inv_marg_u_adj_plus = new double[par->Na];

    #pragma omp for
    for(int i_p = 0; i_p < par->Np; i_p++){
        
        // loop over other outer post-decision states
        for(int i_n = 0; i_n < par->Nn; i_n++){
            
            // a. permanent income and durable stock
            double p = par->grid_p[i_p];
            double n = par->grid_n[i_n];

            // b. initialize at zero
            for(int i_a = 0; i_a < par->Na; i_a++){
                w[i_a] = 0.0;
                q[index::d3(i_p,i_n,i_a,par->Np,par->Nn,par->Na)] = 0.0;
            } // a

            // c. loop over shocks and then end-of-period assets
            for(int ishock = 0; ishock < par->Nshocks; ishock++){
                
                // i. shocks
                double psi_plus = par->psi[ishock];
                double psi_plus_w = par->psi_w[ishock];
                double xi_plus = par->xi[ishock];
                double xi_plus_w = par->xi_w[ishock];

                // ii. next-period income and durables
                double p_plus = trans::p_plus_func(p,psi_plus,par);
                double n_plus = trans::n_plus_func(n,par);

                // iii. prepare interpolators
                int *prep_keep, *prep_adj;
                if(!par->do_simple_wq){
                    prep_keep = linear_interp::interp_3d_prep(par->grid_p,par->grid_n,par->Np,par->Nn,p_plus,n_plus,par->Na);
                    prep_adj = linear_interp::interp_2d_prep(par->grid_p,par->Np,p_plus,par->Na);
                }

                // iv. weight
                double weight = psi_plus_w*xi_plus_w;

                // v. next-period cash-on-hand and total resources
                for(int i_a = 0; i_a < par->Na; i_a++){  
        
                    m_plus[i_a] = trans::m_plus_func(par->grid_a[i_a],p_plus,xi_plus,par);
                    x_plus[i_a] = trans::x_plus_func(m_plus[i_a],n_plus,par);
                
                }
                
                // vi. interpolate
                int index_keep_plus = index::d4(t+1,0,0,0,par->T,par->Np,par->Nn,par->Nm);
                int index_adj_plus = index::d3(t+1,0,0,par->T,par->Np,par->Nx);

                if(par->do_simple_wq){

                    for(int i_a = 0; i_a < par->Na; i_a++){

                        inv_v_keep_plus[i_a] = linear_interp::interp_3d(par->grid_p,par->grid_n,par->grid_m,par->Np,par->Nn,par->Nm,&sol->inv_v_keep[index_keep_plus],
                                                p_plus,n_plus,m_plus[i_a]);
                        inv_v_adj_plus[i_a] = linear_interp::interp_2d(par->grid_p,par->grid_x,par->Np,par->Nx,&sol->inv_v_adj[index_adj_plus],
                                                p_plus,x_plus[i_a]);

                        if(compute_q){
                                                        
                            inv_marg_u_keep_plus[i_a] = linear_interp::interp_3d(par->grid_p,par->grid_n,par->grid_m,par->Np,par->Nn,par->Nm,&sol->inv_marg_u_keep[index_keep_plus],
                                                    p_plus,n_plus,m_plus[i_a]);
                            inv_marg_u_adj_plus[i_a] = linear_interp::interp_2d(par->grid_p,par->grid_x,par->Np,par->Nx,&sol->inv_marg_u_adj[index_adj_plus],
                                                    p_plus,x_plus[i_a]);

                        }

                    } // a

                } else {

                    linear_interp::interp_3d_only_last_vec_mon(prep_keep,par->grid_p,par->grid_n,par->grid_m,par->Np,par->Nn,par->Nm,&sol->inv_v_keep[index_keep_plus],
                                                            p_plus,n_plus,m_plus,inv_v_keep_plus,par->Na);
                    linear_interp::interp_2d_only_last_vec_mon(prep_adj,par->grid_p,par->grid_x,par->Np,par->Nx,&sol->inv_v_adj[index_adj_plus],
                                                            p_plus,x_plus,inv_v_adj_plus,par->Na);

                    if(compute_q){
                        
                        linear_interp::interp_3d_only_last_vec_mon_rep(prep_keep,par->grid_p,par->grid_n,par->grid_m,par->Np,par->Nn,par->Nm,&sol->inv_marg_u_keep[index_keep_plus],
                                                                    p_plus,n_plus,m_plus,inv_marg_u_keep_plus,par->Na);
                        linear_interp::interp_2d_only_last_vec_mon_rep(prep_adj,par->grid_p,par->grid_x,par->Np,par->Nx,&sol->inv_marg_u_adj[index_adj_plus],
                                                                    p_plus,x_plus,inv_marg_u_adj_plus,par->Na);

                    } // compute q
                
                delete[] prep_keep;
                delete[] prep_adj;
                
                }

                // vii. max and accumulate
                if(compute_q){

                    for(int i_a = 0; i_a < par->Na; i_a++){                              
                                                
                        double v_plus, marg_u_plus;
                        bool keep = inv_v_keep_plus[i_a] > inv_v_adj_plus[i_a];
                        if(keep){
                            v_plus = -1.0/inv_v_keep_plus[i_a];
                            marg_u_plus = 1.0/inv_marg_u_keep_plus[i_a];
                        } else {
                            v_plus = -1.0/inv_v_adj_plus[i_a];
                            marg_u_plus = 1.0/inv_marg_u_adj_plus[i_a];
                        }

                        w[i_a] += weight*par->beta*v_plus;
                        q[index::d3(i_p,i_n,i_a,par->Np,par->Nn,par->Na)] += weight*par->beta*par->R*marg_u_plus;

                    } // a

                } else {

                    for(int i_a = 0; i_a < par->Na; i_a++){
                        w[i_a] += weight*par->beta*(-1.0/MAX(inv_v_keep_plus[i_a],inv_v_adj_plus[i_a]));
                    } // a
                
                } // compute q

            } // shock

            // d. transform post decision value function
            for(int i_a = 0; i_a < par->Na; i_a++){
                inv_w[index::d3(i_p,i_n,i_a,par->Np,par->Nn,par->Na)] = -1.0/w[i_a];
            } // a

        } // n
    } // p

        // clean
        delete[] m_plus;
        delete[] x_plus;
        delete[] w;
        delete[] inv_v_keep_plus;
        delete[] inv_marg_u_keep_plus;
        delete[] inv_v_adj_plus;
        delete[] inv_marg_u_adj_plus;

    } // parallel

} // compute_wq

} // namespace