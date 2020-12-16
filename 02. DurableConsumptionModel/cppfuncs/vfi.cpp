#ifndef MAIN
#include "header.cpp"
#endif

/////////////////////
// value-of-choice //
/////////////////////

double value_of_choice_vfi(int t, double c, double d, double p, double x, sol_struct *sol, par_struct *par)
{
    
    // a. end-of-period assets
    double a = x-c-d;
    
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
        double n_plus = trans::n_plus_func(d,par);
        double m_plus = trans::m_plus_func(a,p_plus,xi,par);
        double x_plus = trans::x_plus_func(m_plus,n_plus,par);
                
        // iii. weight
        double weight = psi_w*xi_w;
        
        // iv. update
        double inv_v_plus_keep_now = linear_interp::interp_3d(
            par->grid_p,par->grid_n,par->grid_m,
            par->Np,par->Nn,par->Nm,
            &sol->inv_v_keep[(t+1)*par->Np*par->Nn*par->Nm],
            p_plus,n_plus,m_plus);

        double inv_v_plus_adj_now = linear_interp::interp_2d(
            par->grid_p,par->grid_x,
            par->Np,par->Nx,
            &sol->inv_v_adj[(t+1)*par->Np*par->Nx],
            p_plus,x_plus);
        
        double v_plus_now = -HUGE_VAL;
        if(inv_v_plus_keep_now > inv_v_plus_adj_now && inv_v_plus_keep_now > 0){
            v_plus_now = -1.0/inv_v_plus_keep_now;
        } else if(inv_v_plus_adj_now > 0){
            v_plus_now = -1.0/inv_v_plus_adj_now;
        }
        w += weight*par->beta*v_plus_now;

    }

    double v = utility::func(c,d,par) + w;
    
    // c. total value
    return v;

}

//////////
// keep //
//////////

double obj_vfi_keep_gs(double c, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;

    double d = solver_data->n;
    double p = solver_data->p;
    double x = solver_data->m+solver_data->n;
    par_struct *par = solver_data->par;
    sol_struct *sol = solver_data->sol;
    int t = par->t;

    return -value_of_choice_vfi(t,c,d,p,x,sol,par);

}

double obj_vfi_keep(unsigned n, const double *choices, double *grad, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;
    double c = choices[0];
    double d = solver_data->n;
    double p = solver_data->p;
    double x = solver_data->m+solver_data->n;
    par_struct *par = solver_data->par;
    sol_struct *sol = solver_data->sol;
    int t = par->t;

    // value of choice
    double obj = -value_of_choice_vfi(t,c,d,p,x,sol,par);

    // gradient
    if(grad){
        double forward = -value_of_choice_vfi(t,c+EPS,d,p,x,sol,par);
        grad[0] = (forward - obj)/EPS;
    }

    return obj;

}

double ineq_con_vfi_keep(unsigned n, const double *choices, double *grad, void *solver_data_in)
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

double obj_vfi_adj(unsigned n, const double *choices, double *grad, void *solver_data_in)
{

    solver_struct *solver_data = (solver_struct *) solver_data_in;
    double d  = choices[0];
    double c  = choices[1];
    double p = solver_data->p;
    double x = solver_data->x;
    par_struct *par  = solver_data->par;
    sol_struct *sol  = solver_data->sol;
    int t = par->t;

    // value of choice
    double obj = -value_of_choice_vfi(t,c,d,p,x,sol,par);

    // gradient
    if(grad){
        double forward_d = -value_of_choice_vfi(t,c,d+EPS,p,x,sol,par);
        grad[0] = (forward_d - obj)/EPS;
        double forward_c = -value_of_choice_vfi(t,c+EPS,d,p,x,sol,par);
        grad[1] = (forward_c - obj)/EPS;
    }

    return obj;

}

double ineq_con_vfi_adj(unsigned n, const double *choices, double *grad, void *solver_data_in)
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

EXPORT void solve_vfi_keep(par_struct *par, sol_struct *sol)
{
    
    // unpack
    double *inv_v = &sol->inv_v_keep[par->t*par->Np*par->Nn*par->Nm];
    double *c = &sol->c_keep[par->t*par->Np*par->Nn*par->Nm];

    // keep: loop over outer states
    #pragma omp parallel num_threads(par->cppthreads)
    {
    
    double lb[1], ub[1], choices[1];
    solver_struct* solver_data = new solver_struct;
    solver_data->par = par;
    solver_data->sol = sol;
    auto opt = nlopt_create(NLOPT_LD_MMA, 1);

        // settings
        nlopt_set_min_objective(opt, obj_vfi_keep, solver_data);
        nlopt_set_xtol_rel(opt, 1e-6);
        nlopt_set_maxeval(opt, 200);

        // constraints
        nlopt_add_inequality_constraint(opt, ineq_con_vfi_keep, solver_data, 1e-8);

    #pragma omp for
    for(int i_p = 0; i_p < par->Np; i_p++){
    for(int i_n = 0; i_n < par->Nn; i_n++){
            
        // outer states
        solver_data->p = par->grid_p[i_p];
        solver_data->n = par->grid_n[i_n];

        for(int i_m = 0; i_m < par->Nm; i_m++){
            
            int index = i_p*par->Nn*par->Nm + i_n*par->Nm + i_m;

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
    
    } } // p and n

    delete solver_data;
    nlopt_destroy(opt);
        
    } // parallel

} // solve_keep

EXPORT void solve_vfi_adj(par_struct *par, sol_struct *sol)
{

    // unpack
    double* inv_v = &sol->inv_v_adj[par->t*par->Np*par->Nx];
    double* d = &sol->d_adj[par->t*par->Np*par->Nx];
    double* c = &sol->c_adj[par->t*par->Np*par->Nx];

    // loop over outer states
    #pragma omp parallel num_threads(par->cppthreads)
    {

    double lb[2], ub[2], choices[2];
    solver_struct* solver_data = new solver_struct;
    solver_data->par = par;
    solver_data->sol = sol;
    auto opt = nlopt_create(NLOPT_LD_MMA, 2);

        // settings
        nlopt_set_min_objective(opt, obj_vfi_adj, solver_data);
        nlopt_set_xtol_rel(opt, 1e-6);
        nlopt_set_maxeval(opt, 200);
        
        // constraints
        nlopt_add_inequality_constraint(opt, ineq_con_vfi_adj, solver_data, 1e-8);

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
                d[index] = 0;
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
            d[index] = choices[0];
            c[index] = choices[1];
            inv_v[index] = 1.0/minf;
        
        } // x

    } // p

    delete solver_data;
    nlopt_destroy(opt);

    } // parallel

} // solve_adj