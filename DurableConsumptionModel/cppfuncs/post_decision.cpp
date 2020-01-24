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

void compute_wq_2d(int t, sol_struct *sol, par_struct *par, bool compute_q)
{

    // unpack
    auto inv_w = &sol->inv_w_2d[index::d5(t,0,0,0,0,par->T-1,par->Np,par->Nn,par->Nn,par->Na)];
    auto q = &sol->q_2d[index::d5(t,0,0,0,0,par->T-1,par->Np,par->Nn,par->Nn,par->Na)];

    // loop over outermost post-decision state
    #pragma omp parallel num_threads(par->cppthreads)
    {
    
        // allocate temporary containers
        auto m_plus = new double[par->Na]; // container, same lenght as grid_a
        auto x_full_plus = new double[par->Na];
        auto x_d1_plus = new double[par->Na];
        auto x_d2_plus = new double[par->Na];
        auto w = new double[par->Na]; 
        auto inv_v_keep_plus = new double[par->Na];
        auto inv_marg_u_keep_plus = new double[par->Na];
        auto inv_v_adj_full_plus = new double[par->Na];
        auto inv_marg_u_adj_full_plus = new double[par->Na];
        auto inv_v_adj_d1_plus = new double[par->Na];
        auto inv_marg_u_adj_d1_plus = new double[par->Na];
        auto inv_v_adj_d2_plus = new double[par->Na];
        auto inv_marg_u_adj_d2_plus = new double[par->Na];

    #pragma omp for
    for(int i_p = 0; i_p < par->Np; i_p++){
        
        // loop over other outer post-decision states
        for(int i_n1 = 0; i_n1 < par->Nn; i_n1++){
        for(int i_n2 = 0; i_n2 < par->Nn; i_n2++){

            // a. permanent income and durable stock
            double p = par->grid_p[i_p];
            double n1 = par->grid_n[i_n1];
            double n2 = par->grid_n[i_n2];

            // b. initialize at zero
            for(int i_a = 0; i_a < par->Na; i_a++){
                w[i_a] = 0.0;
                q[index::d4(i_p,i_n1,i_n2,i_a,par->Np,par->Nn,par->Nn,par->Na)] = 0.0;
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
                double n1_plus = trans::n1_plus_func(n1,par);
                double n2_plus = trans::n2_plus_func(n2,par);

                // iii. prepare interpolators
                int* prep_keep = linear_interp::interp_4d_prep( par->grid_p,par->grid_n,par->grid_n,
                                                                par->Np,par->Nn,par->Nn,p_plus,n1_plus,n2_plus,par->Na);
                int* prep_adj_full = linear_interp::interp_2d_prep(par->grid_p,par->Np,p_plus,par->Na);
                int* prep_adj_d1 = linear_interp::interp_3d_prep(par->grid_p,par->grid_n,par->Np,par->Nn,p_plus,n2_plus,par->Na);
                int* prep_adj_d2 = linear_interp::interp_3d_prep(par->grid_p,par->grid_n,par->Np,par->Nn,p_plus,n1_plus,par->Na);
                
                // iv. weight
                double weight = psi_plus_w*xi_plus_w;

                // v. next-period cash-on-hand and total resources
                for(int i_a = 0; i_a < par->Na; i_a++){  
        
                    m_plus[i_a] = trans::m_plus_func(par->grid_a[i_a],p_plus,xi_plus,par);
                    x_full_plus[i_a] = trans::x_plus_full_func(m_plus[i_a],n1_plus,n2_plus,par);
                    x_d1_plus[i_a] = trans::x_plus_d1_func(m_plus[i_a],n1_plus,par);
                    x_d2_plus[i_a] = trans::x_plus_d2_func(m_plus[i_a],n2_plus,par);
                
                }
                
                // vi. interpolate
                int index_keep_plus = index::d5(t+1,0,0,0,0,par->T,par->Np,par->Nn,par->Nn,par->Nm);
                int index_adj_full_plus = index::d3(t+1,0,0,par->T,par->Np,par->Nx);
                int index_adj_d1_plus = index::d4(t+1,0,0,0,par->T,par->Np,par->Nn,par->Nx);
                int index_adj_d2_plus = index::d4(t+1,0,0,0,par->T,par->Np,par->Nn,par->Nx);

                linear_interp::interp_4d_only_last_vec_mon( prep_keep,par->grid_p,par->grid_n,par->grid_n,par->grid_m,
                                                            par->Np,par->Nn,par->Nn,par->Nm,
                                                            &sol->inv_v_keep_2d[index_keep_plus],
                                                            p_plus,n1_plus,n2_plus,m_plus,inv_v_keep_plus,par->Na);

                linear_interp::interp_2d_only_last_vec_mon( prep_adj_full,par->grid_p,par->grid_x,par->Np,par->Nx,
                                                            &sol->inv_v_adj_full_2d[index_adj_full_plus],
                                                            p_plus,x_full_plus,inv_v_adj_full_plus,par->Na);
                                                            
                linear_interp::interp_3d_only_last_vec_mon( prep_adj_d1,par->grid_p,par->grid_n,par->grid_x,par->Np,par->Nn,par->Nx,
                                                            &sol->inv_v_adj_d1_2d[index_adj_d1_plus],
                                                            p_plus,n2_plus,x_d1_plus,inv_v_adj_d1_plus,par->Na);                                                            

                linear_interp::interp_3d_only_last_vec_mon( prep_adj_d2,par->grid_p,par->grid_n,par->grid_x,par->Np,par->Nn,par->Nx,
                                                            &sol->inv_v_adj_d2_2d[index_adj_d2_plus],
                                                            p_plus,n1_plus,x_d2_plus,inv_v_adj_d2_plus,par->Na);   

                if(compute_q){
                    
                    linear_interp::interp_4d_only_last_vec_mon_rep( prep_keep,par->grid_p,par->grid_n,par->grid_n,par->grid_m,
                                                                    par->Np,par->Nn,par->Nn,par->Nm,
                                                                    &sol->inv_marg_u_keep_2d[index_keep_plus],
                                                                    p_plus,n1_plus,n2_plus,m_plus,inv_marg_u_keep_plus,par->Na);

                    linear_interp::interp_2d_only_last_vec_mon_rep( prep_adj_full,par->grid_p,par->grid_x,par->Np,par->Nx,
                                                                    &sol->inv_marg_u_adj_full_2d[index_adj_full_plus],
                                                                    p_plus,x_full_plus,inv_marg_u_adj_full_plus,par->Na);

                    linear_interp::interp_3d_only_last_vec_mon_rep( prep_adj_d1,par->grid_p,par->grid_n,par->grid_x,par->Np,par->Nn,par->Nx,
                                                                    &sol->inv_marg_u_adj_d1_2d[index_adj_d1_plus],
                                                                    p_plus,n2_plus,x_d1_plus,inv_marg_u_adj_d1_plus,par->Na);  

                    linear_interp::interp_3d_only_last_vec_mon_rep( prep_adj_d2,par->grid_p,par->grid_n,par->grid_x,par->Np,par->Nn,par->Nx,
                                                                    &sol->inv_marg_u_adj_d2_2d[index_adj_d2_plus],
                                                                    p_plus,n1_plus,x_d2_plus,inv_marg_u_adj_d2_plus,par->Na);                                                                                                                                        

                } // compute q
                
                delete[] prep_keep;
                delete[] prep_adj_full;
                delete[] prep_adj_d1;
                delete[] prep_adj_d2;
         
                // vii. max and accumulate            
                for(int i_a = 0; i_a < par->Na; i_a++){    
                    
                    double v_plus, marg_u_plus;
                    double inv_v_plus = 0;                                                                
                    bool keep = false;
                    bool adj_full = false;
                    bool adj_d1 = false;
                    bool adj_d2 = false;

                    if(inv_v_keep_plus[i_a] > inv_v_plus){
                        inv_v_plus = inv_v_keep_plus[i_a];
                        keep = true;
                    }

                    if(inv_v_adj_full_plus[i_a] > inv_v_plus){
                        inv_v_plus = inv_v_adj_full_plus[i_a];
                        keep = false;
                        adj_full = true;
                    }


                    if(inv_v_adj_d1_plus[i_a] > inv_v_plus){
                        inv_v_plus = inv_v_adj_d1_plus[i_a];
                        keep = false;
                        adj_full = false;
                        adj_d1 = true;
                    }                        

                    if(inv_v_adj_d2_plus[i_a] > inv_v_plus){
                        inv_v_plus = inv_v_adj_d2_plus[i_a];
                        keep = false;
                        adj_full = false;
                        adj_d1 = false;
                        adj_d2 = true;
                    }  

                    if(keep){
                        v_plus = -1.0/inv_v_keep_plus[i_a];
                        if(compute_q){marg_u_plus = 1.0/inv_marg_u_keep_plus[i_a];}
                    } else if(adj_full){
                        v_plus = -1.0/inv_v_adj_full_plus[i_a];
                        if(compute_q){marg_u_plus = 1.0/inv_marg_u_adj_full_plus[i_a];}
                    } else if(adj_d1){
                        v_plus = -1.0/inv_v_adj_d1_plus[i_a];
                        if(compute_q){marg_u_plus = 1.0/inv_marg_u_adj_d1_plus[i_a];}
                    } else if(adj_d2){
                        v_plus = -1.0/inv_v_adj_d2_plus[i_a];
                        if(compute_q){marg_u_plus = 1.0/inv_marg_u_adj_d2_plus[i_a];}
                    }

                    w[i_a] += weight*par->beta*v_plus;
                    if(compute_q){q[index::d4(i_p,i_n1,i_n2,i_a,par->Np,par->Nn,par->Nn,par->Na)] += weight*par->beta*par->R*marg_u_plus;}

                } // a

            } // shock

            // d. transform post decision value function
            for(int i_a = 0; i_a < par->Na; i_a++){
                inv_w[index::d4(i_p,i_n1,i_n2,i_a,par->Np,par->Nn,par->Nn,par->Na)] = -1.0/w[i_a];
            } // a

        } // n1
        } // n2

    } // p

        // clean
        delete[] m_plus;
        delete[] x_full_plus;
        delete[] x_d1_plus;
        delete[] x_d2_plus;
        delete[] w;
        delete[] inv_v_keep_plus;
        delete[] inv_marg_u_keep_plus;
        delete[] inv_v_adj_full_plus;
        delete[] inv_marg_u_adj_full_plus;
        delete[] inv_v_adj_d1_plus;
        delete[] inv_marg_u_adj_d1_plus;
        delete[] inv_v_adj_d2_plus;
        delete[] inv_marg_u_adj_d2_plus;

    } // parallel

} // compute_wq

} // namespace