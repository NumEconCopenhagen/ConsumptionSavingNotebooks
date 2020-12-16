#ifndef MAIN
#include "header.cpp"
#endif

/////////////////////
// value-of-choice //
/////////////////////

double value_of_choice_vfi_2d(int t, double c, double d1, double d2, double p, double x, sol_struct *sol, par_struct *par)
{
    
    // a. end-of-period assets
    double a = x-c-d1-d2;
    
    // b. continuation value
    double w = 0;
    for(int ishock = 0; ishock < par->Nshocks; ishock++){
            
        // i. shocks
        double psi = par->psi[ishock];
        double psi_w = par->psi_w[ishock];
        double xi = par->xi[ishock];
        double xi_w = par->xi_w[ishock];

        // ii. next-period states
        double p_plus = trans::p_plus_func(p,psi,par);
        double n1_plus = trans::n1_plus_func(d1,par);
        double n2_plus = trans::n2_plus_func(d2,par);
        double m_plus = trans::m_plus_func(a,p_plus,xi,par);
        double x_full_plus = trans::x_plus_full_func(m_plus,n1_plus,n2_plus,par);
        double x_d1_plus = trans::x_plus_d1_func(m_plus,n1_plus,par);
        double x_d2_plus = trans::x_plus_d2_func(m_plus,n2_plus,par);
                
        // iii. weight
        double weight = psi_w*xi_w;
        
        // iv. update
        double inv_v_plus = 0;
        
        // keep
        double inv_v_plus_keep_now = linear_interp::interp_4d(
            par->grid_p,par->grid_n,par->grid_n,par->grid_m,
            par->Np,par->Nn,par->Nn,par->Nm,
            &sol->inv_v_keep_2d[(t+1)*par->Np*par->Nn*par->Nn*par->Nm],
            p_plus,n1_plus,n2_plus,m_plus);
        
        inv_v_plus = MAX(inv_v_plus_keep_now,inv_v_plus);

        // adj full
        double inv_v_plus_adj_full_now = linear_interp::interp_2d(
            par->grid_p,par->grid_x,
            par->Np,par->Nx,
            &sol->inv_v_adj_full_2d[(t+1)*par->Np*par->Nx],
            p_plus,x_full_plus);
        
        inv_v_plus = MAX(inv_v_plus_adj_full_now,inv_v_plus);
        
        // adj d1
        double inv_v_plus_adj_d1_now = linear_interp::interp_3d(
            par->grid_p,par->grid_n,par->grid_x,
            par->Np,par->Nn,par->Nx,
            &sol->inv_v_adj_d1_2d[(t+1)*par->Np*par->Nn*par->Nx],
            p_plus,n2_plus,x_d1_plus);

        inv_v_plus = MAX(inv_v_plus_adj_d1_now,inv_v_plus);

        // adj d2
        double inv_v_plus_adj_d2_now = linear_interp::interp_3d(
            par->grid_p,par->grid_n,par->grid_x,
            par->Np,par->Nn,par->Nx,
            &sol->inv_v_adj_d2_2d[(t+1)*par->Np*par->Nn*par->Nx],
            p_plus,n1_plus,x_d2_plus);

        inv_v_plus = MAX(inv_v_plus_adj_d2_now,inv_v_plus);

        // convert and sum
        double v_plus_now = -HUGE_VAL;
        if(inv_v_plus > 0){
            v_plus_now = -1.0/inv_v_plus;
        }
        w += weight*par->beta*v_plus_now;

    }


    double v = utility::func_2d(c,d1,d2,par) + w;
    
    // c. total value
    return v;

}

//////////
// keep //
//////////

double obj_vfi_2d_keep_gs(double c, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;

    double d1 = solver_data->n1;
    double d2 = solver_data->n2;
    double p = solver_data->p;
    double x = solver_data->m+solver_data->n1+solver_data->n2;
    par_struct *par = solver_data->par;
    sol_struct *sol = solver_data->sol;
    int t = par->t;

    return -value_of_choice_vfi_2d(t,c,d1,d2,p,x,sol,par);

}

double obj_vfi_2d_keep(unsigned n, const double *choices, double *grad, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;
    double c = choices[0];
    double d1 = solver_data->n1;
    double d2 = solver_data->n2;
    double p = solver_data->p;
    double x = solver_data->m+solver_data->n1+solver_data->n2;
    par_struct *par = solver_data->par;
    sol_struct *sol = solver_data->sol;
    int t = par->t;

    // value of choice
    double obj = -value_of_choice_vfi_2d(t,c,d1,d2,p,x,sol,par);

    // gradient
    if(grad){
        double forward = -value_of_choice_vfi_2d(t,c+EPS,d1,d2,p,x,sol,par);
        grad[0] = (forward - obj)/EPS;
    }

    return obj;

}

double ineq_con_vfi_2d_keep(unsigned n, const double *choices, double *grad, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;

    if (grad) {
        grad[0] = 1.0;
    }

    return choices[0] - solver_data->m; // positive if violated

}

/////////
// adj //
/////////

double obj_vfi_2d_adj_full(unsigned n, const double *choices, double *grad, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;
    double d1 = choices[0];
    double d2 = choices[1];
    double c = choices[2];
    double p = solver_data->p;
    double x = solver_data->x;
    par_struct *par = solver_data->par;
    sol_struct *sol = solver_data->sol;
    int t = par->t;

    // value of choice
    double obj = -value_of_choice_vfi_2d(t,c,d1,d2,p,x,sol,par);

    // gradient
    if(grad){
        double forward_d1 = -value_of_choice_vfi_2d(t,c,d1+EPS,d2,p,x,sol,par);
        grad[0] = (forward_d1 - obj)/EPS;
        double forward_d2 = -value_of_choice_vfi_2d(t,c,d1,d2+EPS,p,x,sol,par);
        grad[1] = (forward_d2 - obj)/EPS;        
        double forward_c = -value_of_choice_vfi_2d(t,c+EPS,d1,d2,p,x,sol,par);
        grad[2] = (forward_c - obj)/EPS;
    }

    return obj;

}

double obj_vfi_2d_adj_d1(unsigned n, const double *choices, double *grad, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;
    double d1 = choices[0];
    double d2 = solver_data->n2;
    double c  = choices[1];
    double p = solver_data->p;
    double x = solver_data->x+solver_data->n2;
    par_struct *par = solver_data->par;
    sol_struct *sol = solver_data->sol;
    int t = par->t;

    // value of choice
    double obj = -value_of_choice_vfi_2d(t,c,d1,d2,p,x,sol,par);

    // gradient
    if(grad){
        double forward_d1 = -value_of_choice_vfi_2d(t,c,d1+EPS,d2,p,x,sol,par);
        grad[0] = (forward_d1 - obj)/EPS;
        double forward_c = -value_of_choice_vfi_2d(t,c+EPS,d1,d2,p,x,sol,par);
        grad[1] = (forward_c - obj)/EPS;
    }

    return obj;

}

double obj_vfi_2d_adj_d2(unsigned n, const double *choices, double *grad, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;
    double d1 = solver_data->n1;
    double d2 = choices[0];
    double c  = choices[1];
    double p = solver_data->p;
    double x = solver_data->x+solver_data->n1;
    par_struct *par = solver_data->par;
    sol_struct *sol = solver_data->sol;
    int t = par->t;

    // value of choice
    double obj = -value_of_choice_vfi_2d(t,c,d1,d2,p,x,sol,par);

    // gradient
    if(grad){
        double forward_d2 = -value_of_choice_vfi_2d(t,c,d1,d2+EPS,p,x,sol,par);
        grad[0] = (forward_d2 - obj)/EPS;
        double forward_c = -value_of_choice_vfi_2d(t,c+EPS,d1,d2,p,x,sol,par);
        grad[1] = (forward_c - obj)/EPS;
    }

    return obj;

}

double ineq_con_vfi_2d_adj_full(unsigned n, const double *choices, double *grad, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;

    if (grad) {
        grad[0] = 1.0;
        grad[1] = 1.0;
        grad[2] = 1.0;
    }

    return choices[0] + choices[1] + choices[2] - solver_data->x; // positive if violated
    
}

double ineq_con_vfi_2d_adj_d1(unsigned n, const double *choices, double *grad, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;

    if (grad) {
        grad[0] = 1.0;
        grad[1] = 1.0;
    }

    return choices[0] + choices[1] - solver_data->x; // positive if violated
    
}

double ineq_con_vfi_2d_adj_d2(unsigned n, const double *choices, double *grad, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;

    if (grad) {
        grad[0] = 1.0;
        grad[1] = 1.0;
    }

    return choices[0] + choices[1] - solver_data->x; // positive if violated
    
}

//////////////
// gateways //
//////////////

// find c given p,n1,n2,m
EXPORT void solve_vfi_2d_keep(par_struct *par, sol_struct *sol)
{
    
    // unpack
    int index_t = par->t*par->Np*par->Nn*par->Nn*par->Nm;
    double *inv_v = &sol->inv_v_keep_2d[index_t];
    double *c = &sol->c_keep_2d[index_t];

    // loop over outer states
    #pragma omp parallel num_threads(par->cppthreads)
    {
    
    double lb[1], ub[1], choices[1];
    solver_struct* solver_data = new solver_struct;
    solver_data->par = par;
    solver_data->sol = sol;
    auto opt = nlopt_create(NLOPT_LD_MMA, 1);

        // settings
        nlopt_set_min_objective(opt, obj_vfi_2d_keep, solver_data);
        nlopt_set_xtol_rel(opt, 1e-6);
        nlopt_set_maxeval(opt, 200);

        // constraints
        nlopt_add_inequality_constraint(opt, ineq_con_vfi_2d_keep, solver_data, 1e-8);

    #pragma omp for
    for(int i_p = 0; i_p < par->Np; i_p++){
    for(int i_n1 = 0; i_n1 < par->Nn; i_n1++){
    for(int i_n2 = 0; i_n2 < par->Nn; i_n2++){
        
        // outer states
        solver_data->p = par->grid_p[i_p];
        solver_data->n1 = par->grid_n[i_n1];
        solver_data->n2 = par->grid_n[i_n2];

        for(int i_m = 0; i_m < par->Nm; i_m++){
            
            int index = i_p*par->Nn*par->Nn*par->Nm + i_n1*par->Nn*par->Nm + i_n2*par->Nm + i_m;

            // a. cash-on-hand
            solver_data->m = par->grid_m[i_m];
                
            if(i_m == 0){
                c[index] = 0;
                inv_v[index] = 0;
                continue;
            } else if(i_m == 1){
                choices[0] = solver_data->m/2;
            } 
            
            // b. optimal choice
            lb[0] = 0;
            ub[0] = solver_data->m;
            nlopt_set_lower_bounds(opt, lb);
            nlopt_set_upper_bounds(opt, ub);

            double minf;
            int flag = nlopt_optimize(opt, choices, &minf);

            // c. optimal value
            c[index] = choices[0];
            inv_v[index] = 1.0/minf;            

        } // m
    
    } } } // p and n1, n2

    delete solver_data;
    nlopt_destroy(opt);
        
    } // parallel

} // solve_keep

// find c,d1,d2 given p,x
EXPORT void solve_vfi_2d_adj_full(par_struct *par, sol_struct *sol)
{

    // unpack
    int index_t = par->t*par->Np*par->Nx;
    double* inv_v = &sol->inv_v_adj_full_2d[index_t];
    double* d1 = &sol->d1_adj_full_2d[index_t];
    double* d2 = &sol->d2_adj_full_2d[index_t];
    double* c = &sol->c_adj_full_2d[index_t];

    // loop over outer states
    #pragma omp parallel num_threads(par->cppthreads)
    {

    double lb[3], ub[3], choices[3];
    solver_struct* solver_data = new solver_struct;
    solver_data->par = par;
    solver_data->sol = sol;
    auto opt = nlopt_create(NLOPT_LD_MMA, 3);

        // settings
        nlopt_set_min_objective(opt, obj_vfi_2d_adj_full, solver_data);
        nlopt_set_xtol_rel(opt, 1e-6);
        nlopt_set_maxeval(opt, 200);

        // constraints
        nlopt_add_inequality_constraint(opt, ineq_con_vfi_2d_adj_full, solver_data, 1e-8);

    #pragma omp for
    for(int i_p = 0; i_p < par->Np; i_p++){
            
        // outer states
        solver_data->p = par->grid_p[i_p];

        // loop over x state
        for(int i_x = 0; i_x < par->Nx; i_x++){
            
            int index = i_p*par->Nx+i_x;
            
            // a. cash-on-hand
            solver_data->x = par->grid_x[i_x];
            
            if(i_x == 0){
                d1[index] = 0;
                d2[index] = 0;
                c[index] = 0;
                inv_v[index] = 0;
                continue;
            } else if(i_x == 1){
                choices[0] = solver_data->x/4;
                choices[1] = solver_data->x/4;
                choices[2] = solver_data->x/4;
            }
    
            // b. optimal choice
            lb[0] = 0;
            lb[1] = 0;
            lb[2] = 0;
            ub[0] = solver_data->x;
            ub[1] = solver_data->x;
            ub[2] = solver_data->x;
            nlopt_set_lower_bounds(opt, lb);
            nlopt_set_upper_bounds(opt, ub);

            double minf;
            int flag = nlopt_optimize(opt, choices, &minf);

            // c. optimal value
            d1[index] = choices[0];
            d2[index] = choices[1];
            c[index] = choices[2];
            inv_v[index] = 1/minf;
        
        } // x

    } // p

    delete solver_data;
    nlopt_destroy(opt);

    } // parallel

} // solve_adj_full


// find c,d1 given p,n2,x
EXPORT void solve_vfi_2d_adj_d1(par_struct *par, sol_struct *sol)
{

    int index_t = par->t*par->Np*par->Nn*par->Nx;
    double* inv_v = &sol->inv_v_adj_d1_2d[index_t];
    double* d1 = &sol->d1_adj_d1_2d[index_t];
    double* c = &sol->c_adj_d1_2d[index_t];

    // loop over outer states
    #pragma omp parallel num_threads(par->cppthreads)
    {

    double lb[2], ub[2], choices[2];
    solver_struct* solver_data = new solver_struct;
    solver_data->par = par;
    solver_data->sol = sol;
    auto opt = nlopt_create(NLOPT_LD_MMA, 2);

        // settings
        nlopt_set_min_objective(opt, obj_vfi_2d_adj_d1, solver_data);
        nlopt_set_xtol_rel(opt, 1e-6);
        nlopt_set_maxeval(opt, 200);

        // constraints
        nlopt_add_inequality_constraint(opt, ineq_con_vfi_2d_adj_d1, solver_data, 1e-8);

    #pragma omp for
    for(int i_p = 0; i_p < par->Np; i_p++){

    for(int i_n2 = 0; i_n2 < par->Nn; i_n2++){
            
        // outer states
        solver_data->p = par->grid_p[i_p];
        solver_data->n2 = par->grid_n[i_n2];

        // loop over x state
        for(int i_x = 0; i_x < par->Nx; i_x++){
            
            int index = i_p*par->Nn*par->Nx + i_n2*par->Nx + i_x;
            
            // a. cash-on-hand
            solver_data->x = par->grid_x[i_x];
            
            if(i_x == 0){
                d1[index] = 0;
                c[index] = 0;
                inv_v[index] = 0;
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
            d1[index] = choices[0];
            c[index] = choices[1];
            inv_v[index] = 1.0/minf;
        
        } // x

    } // n2
    } // p

    delete solver_data;
    nlopt_destroy(opt);

    } // parallel

} // solve_adj_d1

// find c,d2 given p,n1,x
EXPORT void solve_vfi_2d_adj_d2(par_struct *par, sol_struct *sol)
{

    // unpack
    int index_t = par->t*par->Np*par->Nn*par->Nx;
    double* inv_v = &sol->inv_v_adj_d2_2d[index_t];
    double* d2 = &sol->d2_adj_d2_2d[index_t];
    double* c = &sol->c_adj_d2_2d[index_t];

    // loop over outer states
    #pragma omp parallel num_threads(par->cppthreads)
    {

    double lb[2], ub[2], choices[2];
    solver_struct* solver_data = new solver_struct;
    solver_data->par = par;
    solver_data->sol = sol;
    auto opt = nlopt_create(NLOPT_LD_MMA, 2);

        // settings
        nlopt_set_min_objective(opt, obj_vfi_2d_adj_d2, solver_data);
        nlopt_set_xtol_rel(opt, 1e-6);
        nlopt_set_maxeval(opt, 200);

        // constraints
        nlopt_add_inequality_constraint(opt, ineq_con_vfi_2d_adj_d2, solver_data, 1e-8);

    #pragma omp for
    for(int i_p = 0; i_p < par->Np; i_p++){

    for(int i_n1 = 0; i_n1 < par->Nn; i_n1++){

        // outer states
        solver_data->p = par->grid_p[i_p];
        solver_data->n1 = par->grid_n[i_n1];

        // loop over x state
        for(int i_x = 0; i_x < par->Nx; i_x++){
            
            int index = i_p*par->Nn*par->Nx + i_n1*par->Nx + i_x;
            
            // a. cash-on-hand
            solver_data->x = par->grid_x[i_x];
            
            if(i_x == 0){
                d2[index] = 0;
                c[index] = 0;
                inv_v[index] = 0;
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
            d2[index] = choices[0];
            c[index] = choices[1];
            inv_v[index] = 1/minf;
        
        } // x

    } // n1
    } // p

    delete solver_data;
    nlopt_destroy(opt);

    } // parallel

} // solve_adj_d2