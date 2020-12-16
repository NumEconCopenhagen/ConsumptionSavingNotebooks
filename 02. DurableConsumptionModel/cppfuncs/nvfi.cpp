#ifndef MAIN
#include "header.cpp"
#endif

EXPORT void compute_wq_nvfi(par_struct *par, sol_struct *sol)
{
    post_decision::compute_wq(par->t,sol,par,false);
}

//////////
// keep //
//////////

double obj_nvfi_keep(double c, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;
    
    auto par = solver_data->par;
    auto inv_w = solver_data->inv_w;

    auto  n = solver_data->n;
    auto  m = solver_data->m;

    // a. end-of-period assets
    double a = m-c;
    
    // b. continuation value
    double w = -1.0/linear_interp::interp_1d(par->grid_a,par->Na,inv_w,a);

    // c. total value
    double value_of_choice = utility::func(c,n,par) + w;

    return -value_of_choice; // we are minimizing

}

EXPORT void solve_nvfi_keep(par_struct *par, sol_struct *sol)
{

    // unpack
    int t = par->t;
    int index_keep_t = index::d4(t,0,0,0,par->T,par->Np,par->Nn,par->Nm);
    double *inv_v = &sol->inv_v_keep[index_keep_t];
    double *inv_marg_u = &sol->inv_marg_u_keep[index_keep_t];
    double *c = &sol->c_keep[index_keep_t];

    // loop over outer states
    #pragma omp parallel num_threads(par->cppthreads)
    {

    double lb[1], ub[1], choices[1];

    solver_struct* solver_data = new solver_struct;
    solver_data->par = par;

    #pragma omp for
    for(int i_p = 0; i_p < par->Np; i_p++){
        for(int i_n = 0; i_n < par->Nn; i_n++){
            
            // outer states
            double n = par->grid_n[i_n];
            solver_data->n = n;

            // loop over m state
            for(int i_m = 0; i_m < par->Nm; i_m++){
                
                int index = index::d3(i_p,i_n,i_m,par->Np,par->Nn,par->Nm);
                solver_data->inv_w = &sol->inv_w[index::d4(t,i_p,i_n,0,par->T-1,par->Np,par->Nn,par->Na)];

                // a. cash-on-hand
                double m = par->grid_m[i_m];
                solver_data->m = m;

                if(i_m == 0){
                    c[index] = 0;
                    inv_v[index] = 0;
                    if(par->do_marg_u){
                        inv_marg_u[index] = 0; 
                    }
                    continue;
                } else if(i_m == 1){
                    choices[0] = solver_data->m*0.99;
                }
                                
                // b. optimal choice
                double c_low = MIN(m/2,1e-8);
                double c_high = m;

                c[index] = golden_section_search(c_low,c_high,par->tol,solver_data,obj_nvfi_keep); 
                double v = -obj_nvfi_keep(c[index],solver_data);
                
                // c. optimal value
                inv_v[index] = -1.0/v;
                if(par->do_marg_u){
                    inv_marg_u[index] = 1.0/utility::marg_func(c[index],n,par);
                }

            } // m
        } // n
    } // p

        delete solver_data;

    } // parallel

} // solve_keep

/////////
// adj //
/////////

double obj_nvfi_adj(double d, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;
    
    auto par = solver_data->par;
    auto inv_v_keep = solver_data->inv_v_keep;

    auto  x = solver_data->x;

    // a. cash-on-hand
    double m = x-d;

    // b. durables
    double n = d;
    
    // c. value-of-choice
    return -linear_interp::interp_2d(par->grid_n,par->grid_m,par->Nn,par->Nm,inv_v_keep,n,m); // we are minimizing

}

EXPORT void solve_nvfi_adj(par_struct *par, sol_struct *sol)
{

    // unpack
    int t = par->t;
    int index_adj_t = index::d3(t,0,0,par->T,par->Np,par->Nx);
    double *inv_v = &sol->inv_v_adj[index_adj_t];
    double *inv_marg_u = &sol->inv_marg_u_adj[index_adj_t];
    double *d = &sol->d_adj[index_adj_t];
    double *c = &sol->c_adj[index_adj_t];

    // loop over outer states
    #pragma omp parallel num_threads(par->cppthreads)
    {

    double lb[1], ub[1], choices[1];

    solver_struct* solver_data = new solver_struct;
    solver_data->par = par;

    #pragma omp for
    for(int i_p = 0; i_p < par->Np; i_p++){
            
        // loop over x state
        for(int i_x = 0; i_x < par->Nx; i_x++){
            
            int index_adj = index::d2(i_p,i_x,par->Np,par->Nx);
            int index_keep = index::d4(t,i_p,0,0,par->T,par->Np,par->Nn,par->Nm);
            solver_data->inv_v_keep = &sol->inv_v_keep[index_keep];
            
            // a. cash-on-hand
            double x = par->grid_x[i_x];
            solver_data->x = x;

            if(i_x == 0){
                d[index_adj] = 0;
                c[index_adj] = 0;
                inv_v[index_adj] = 0;
                if(par->do_marg_u){
                    inv_marg_u[index_adj] = 0; 
                }
                continue;
            } else if(i_x == 1){
                choices[0] = solver_data->x/3;
            }

            // b. optimal choice
            double d_low = MIN(x/2,1e-8);
            double d_high = MIN(x,par->n_max);

            d[index_adj] =  golden_section_search(d_low,d_high,par->tol,solver_data,obj_nvfi_adj); 
            inv_v[index_adj] = -obj_nvfi_adj(d[index_adj],solver_data);

            // c. optimal value
            double m = x - d[index_adj];
            c[index_adj] = linear_interp::interp_2d(par->grid_n,par->grid_m,par->Nn,par->Nm,&sol->c_keep[index_keep],d[index_adj],m);
            if(par->do_marg_u){
                inv_marg_u[index_adj] = 1.0/utility::marg_func(c[index_adj],d[index_adj],par);
            }

        } // x

    } // p

        delete solver_data;
        
    } // parallel

} // solve_adj