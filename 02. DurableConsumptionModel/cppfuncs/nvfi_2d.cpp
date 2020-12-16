#ifndef MAIN
#include "header.cpp"
#endif

EXPORT void compute_wq_nvfi_2d(par_struct *par, sol_struct *sol)
{
    post_decision::compute_wq_2d(par->t,sol,par,false);
}

//////////
// keep //
//////////

double obj_nvfi_2d_keep(double c, void *solver_data_in)
{

    // a. unpack
    solver_struct *solver_data = (solver_struct *) solver_data_in;
    
    auto par = solver_data->par;
    auto inv_w = solver_data->inv_w;

    auto n1 = solver_data->n1;
    auto n2 = solver_data->n2;
    auto m = solver_data->m;

    // a. end-of-period assets
    double a = m-c;
    
    // b. continuation value
    double w = -1.0/linear_interp::interp_1d(par->grid_a,par->Na,inv_w,a);

    // c. total value
    double value_of_choice = utility::func_2d(c,n1,n2,par) + w;

    return -value_of_choice; // we are minimizing

}

EXPORT void solve_nvfi_2d_keep(par_struct *par, sol_struct *sol)
{

    // unpack
    int t = par->t;
    int index_t = index::d5(t,0,0,0,0,par->T,par->Np,par->Nn,par->Nn,par->Nm);
    double *inv_v = &sol->inv_v_keep_2d[index_t];
    double *inv_marg_u = &sol->inv_marg_u_keep_2d[index_t];
    double *c = &sol->c_keep_2d[index_t];

    // loop over outer states
    #pragma omp parallel num_threads(par->cppthreads)
    {

    double lb[1], ub[1], choices[1];

    solver_struct* solver_data = new solver_struct;
    solver_data->par = par;

    #pragma omp for
    for(int i_p = 0; i_p < par->Np; i_p++){

        for(int i_n1 = 0; i_n1 < par->Nn; i_n1++){
        for(int i_n2 = 0; i_n2 < par->Nn; i_n2++){
            
            // outer states
            double n1 = par->grid_n[i_n1];
            double n2 = par->grid_n[i_n2];
            solver_data->n1 = n1;
            solver_data->n2 = n2;

            // loop over m state
            for(int i_m = 0; i_m < par->Nm; i_m++){
                
                int index = index::d4(i_p,i_n1,i_n2,i_m,par->Np,par->Nn,par->Nn,par->Nm);
                solver_data->inv_w = &sol->inv_w_2d[index::d5(t,i_p,i_n1,i_n2,0,par->T-1,par->Np,par->Nn,par->Nn,par->Na)];

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

                c[index] = golden_section_search(c_low,c_high,par->tol,solver_data,obj_nvfi_2d_keep); 
                double v = -obj_nvfi_2d_keep(c[index],solver_data);
                
                // c. optimal value
                inv_v[index] = -1.0/v;
                if(par->do_marg_u){
                    inv_marg_u[index] = 1.0/utility::marg_func_2d(c[index],n1,n2,par);
                }

            } // m
        } } // n1, n2
    } // p

        delete solver_data;

    } // parallel

} // solve_keep

/////////
// adj //
/////////

double obj_nvfi_2d_adj_full(unsigned n, const double *choices, double *grad, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;
    double d1 = choices[0];    
    double d2 = choices[1];    

    auto par = solver_data->par;
    auto inv_v_keep = solver_data->inv_v_keep;

    auto x = solver_data->x;

    // a. cash-on-hand
    double m = x-d1-d2;

    // b. durables
    double n1 = d1;
    double n2 = d2;
    
    // c. value-of-choice
    double obj = -linear_interp::interp_3d(par->grid_n,par->grid_n,par->grid_m,par->Nn,par->Nn,par->Nm,inv_v_keep,n1,n2,m);

    // d. gradient
    if(grad){
        double forward_d1 = -linear_interp::interp_3d(par->grid_n,par->grid_n,par->grid_m,par->Nn,par->Nn,par->Nm,inv_v_keep,n1+EPS,n2,m-EPS);
        grad[0] = (forward_d1 - obj)/EPS;
        double forward_d2 = -linear_interp::interp_3d(par->grid_n,par->grid_n,par->grid_m,par->Nn,par->Nn,par->Nm,inv_v_keep,n1,n2+EPS,m-EPS);
        grad[1] = (forward_d2 - obj)/EPS;
    }

    return obj;

}

double ineq_obj_nvfi_2d_adj_full(unsigned n, const double *choices, double *grad, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;

    if (grad) {
        grad[0] = 1.0;
        grad[1] = 1.0;
    }

    return choices[0] + choices[1] - solver_data->x; // positive if violated
    
}

double obj_nvfi_2d_adj_d1(double d1, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;
    
    auto par = solver_data->par;
    auto inv_v_keep = solver_data->inv_v_keep;

    auto  x = solver_data->x;

    // a. cash-on-hand
    double m = x-d1;

    // b. durables
    double n1 = d1;
    double n2 = solver_data->n2;
    
    // c. value-of-choice
    return -linear_interp::interp_3d(par->grid_n,par->grid_n,par->grid_m,par->Nn,par->Nn,par->Nm,inv_v_keep,n1,n2,m); // we are minimizing

}

double obj_nvfi_2d_adj_d2(double d2, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;
    
    auto par = solver_data->par;
    auto inv_v_keep = solver_data->inv_v_keep;

    auto  x = solver_data->x;

    // a. cash-on-hand
    double m = x-d2;

    // b. durables
    double n1 = solver_data->n1;
    double n2 = d2;
    
    // c. value-of-choice
    return -linear_interp::interp_3d(par->grid_n,par->grid_n,par->grid_m,par->Nn,par->Nn,par->Nm,inv_v_keep,n1,n2,m); // we are minimizing

}

EXPORT void solve_nvfi_2d_adj_full(par_struct *par, sol_struct *sol)
{

    // unpack
    int t = par->t;
    int index_t = index::d3(t,0,0,par->T,par->Np,par->Nx);
    double *inv_v = &sol->inv_v_adj_full_2d[index_t];
    double *inv_marg_u = &sol->inv_marg_u_adj_full_2d[index_t];
    double *d1 = &sol->d1_adj_full_2d[index_t];
    double *d2 = &sol->d2_adj_full_2d[index_t];
    double *c = &sol->c_adj_full_2d[index_t];

    // loop over outer states
    #pragma omp parallel num_threads(1) //num_threads(par->cppthreads)
    {

    double lb[2], ub[2], choices[2];

    solver_struct* solver_data = new solver_struct;
    solver_data->par = par;

    auto opt = nlopt_create(NLOPT_LD_MMA, 2);

        // settings
        nlopt_set_min_objective(opt, obj_nvfi_2d_adj_full, solver_data);
        nlopt_set_xtol_rel(opt, 1e-6);
        nlopt_set_maxeval(opt, 200);

        // constraints
        nlopt_add_inequality_constraint(opt, ineq_obj_nvfi_2d_adj_full, solver_data, 1e-8);

    #pragma omp for
    for(int i_p = 0; i_p < par->Np; i_p++){
        for(int i_x = 0; i_x < par->Nx; i_x++){

            int index_adj = index::d2(i_p,i_x,par->Np,par->Nx);
            int index_keep = index::d5(t,i_p,0,0,0,par->T,par->Np,par->Nn,par->Nn,par->Nm);
            solver_data->inv_v_keep = &sol->inv_v_keep_2d[index_keep];
            
            // a. cash-on-hand
            double x = par->grid_x[i_x];
            solver_data->x = x;

            if(i_x == 0){
                d1[index_adj] = 0;
                d2[index_adj] = 0;
                c[index_adj] = 0;
                inv_v[index_adj] = 0;
                if(par->do_marg_u){
                    inv_marg_u[index_adj] = 0; 
                }
                continue;
            } else if(i_x == 1){
                choices[0] = solver_data->x/3;
                choices[1] = solver_data->x/3;
            }

            // b. optimal choice
            lb[0] = 0;
            lb[1] = 0;
            ub[0] = solver_data->x;
            ub[1] = solver_data->x;
            nlopt_set_lower_bounds(opt, lb);
            nlopt_set_upper_bounds(opt, ub);

            double minf;
            int flag = nlopt_optimize(opt, choices, &minf);
            
            // c. optimal value
            double m = x - d1[index_adj] - d2[index_adj];
            d1[index_adj] = choices[0];
            d2[index_adj] = choices[1];            
            c[index_adj] = linear_interp::interp_3d(par->grid_n,par->grid_n,par->grid_m,par->Nn,par->Nn,par->Nm,
                                                    &sol->c_keep_2d[index_keep],d1[index_adj],d2[index_adj],m);
            inv_v[index_adj] = -minf;
            
            if(par->do_marg_u){
                inv_marg_u[index_adj] = 1.0/utility::marg_func_2d(c[index_adj],d1[index_adj],d2[index_adj],par);
            }

        } // x
    } // p

        delete solver_data;
        
    } // parallel

} // solve_adj_full

EXPORT void solve_nvfi_2d_adj_d1(par_struct *par, sol_struct *sol)
{

    // unpack
    int t = par->t;
    int index_t = index::d4(t,0,0,0,par->T,par->Np,par->Nn,par->Nx);
    double *inv_v = &sol->inv_v_adj_d1_2d[index_t];
    double *inv_marg_u = &sol->inv_marg_u_adj_d1_2d[index_t];
    double *d1 = &sol->d1_adj_d1_2d[index_t];
    double *c = &sol->c_adj_d1_2d[index_t];

    // loop over outer states
    #pragma omp parallel num_threads(par->cppthreads)
    {

    double lb[1], ub[1], choices[1];

    solver_struct* solver_data = new solver_struct;
    solver_data->par = par;

    #pragma omp for
    for(int i_p = 0; i_p < par->Np; i_p++){
            
        for(int i_n2 = 0; i_n2 < par->Nn; i_n2++){

            double n2 = par->grid_n[i_n2];
            solver_data->n2 = n2;

        // loop over x state
        for(int i_x = 0; i_x < par->Nx; i_x++){

            int index_adj = index::d3(i_p,i_n2,i_x,par->Np,par->Nn,par->Nx);
            int index_keep = index::d5(t,i_p,0,0,0,par->T,par->Np,par->Nn,par->Nn,par->Nm);
            solver_data->inv_v_keep = &sol->inv_v_keep_2d[index_keep];
            
            // a. cash-on-hand
            double x = par->grid_x[i_x];
            solver_data->x = x;

            if(i_x == 0){
                d1[index_adj] = 0;
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

            d1[index_adj] = golden_section_search(d_low,d_high,par->tol,solver_data,obj_nvfi_2d_adj_d1); 
            inv_v[index_adj] = -obj_nvfi_2d_adj_d1(d1[index_adj],solver_data);

            // c. optimal value
            double m = x - d1[index_adj];
            c[index_adj] = linear_interp::interp_3d(par->grid_n,par->grid_n,par->grid_m,par->Nn,par->Nn,par->Nm,
                                                    &sol->c_keep_2d[index_keep],d1[index_adj],n2,m);

            if(par->do_marg_u){
                inv_marg_u[index_adj] = 1.0/utility::marg_func_2d(c[index_adj],d1[index_adj],n2,par);
            }

        } // x
        } // n

    } // p

        delete solver_data;
        
    } // parallel

} // solve_adj_d1

EXPORT void solve_nvfi_2d_adj_d2(par_struct *par, sol_struct *sol)
{

    // unpack
    int t = par->t;
    int index_t = index::d4(t,0,0,0,par->T,par->Np,par->Nn,par->Nx);
    double *inv_v = &sol->inv_v_adj_d2_2d[index_t];
    double *inv_marg_u = &sol->inv_marg_u_adj_d2_2d[index_t];
    double *d2 = &sol->d2_adj_d2_2d[index_t];
    double *c = &sol->c_adj_d2_2d[index_t];

    // loop over outer states
    #pragma omp parallel num_threads(par->cppthreads)
    {

    double lb[1], ub[1], choices[1];

    solver_struct* solver_data = new solver_struct;
    solver_data->par = par;

    #pragma omp for
    for(int i_p = 0; i_p < par->Np; i_p++){
            
        for(int i_n1 = 0; i_n1 < par->Nn; i_n1++){

            double n1 = par->grid_n[i_n1];
            solver_data->n1 = n1;

        // loop over x state
        for(int i_x = 0; i_x < par->Nx; i_x++){

            int index_adj = index::d3(i_p,i_n1,i_x,par->Np,par->Nn,par->Nx);
            int index_keep = index::d5(t,i_p,0,0,0,par->T,par->Np,par->Nn,par->Nn,par->Nm);
            solver_data->inv_v_keep = &sol->inv_v_keep_2d[index_keep];
            
            // a. cash-on-hand
            double x = par->grid_x[i_x];
            solver_data->x = x;

            if(i_x == 0){
                d2[index_adj] = 0;
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

            d2[index_adj] =  golden_section_search(d_low,d_high,par->tol,solver_data,obj_nvfi_2d_adj_d2); 
            inv_v[index_adj] = -obj_nvfi_2d_adj_d2(d2[index_adj],solver_data);

            // c. optimal value
            double m = x - d2[index_adj];
            c[index_adj] = linear_interp::interp_3d(par->grid_n,par->grid_n,par->grid_m,par->Nn,par->Nn,par->Nm,
                                                    &sol->c_keep_2d[index_keep],n1,d2[index_adj],m);

            if(par->do_marg_u){
                inv_marg_u[index_adj] = 1.0/utility::marg_func_2d(c[index_adj],n1,d2[index_adj],par);
            }

        } // x
        } // n1

    } // p

        delete solver_data;
        
    } // parallel

} // solve_adj_d2
