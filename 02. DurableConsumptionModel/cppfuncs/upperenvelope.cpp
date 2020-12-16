#ifndef MAIN
#define UPPERENVELOPE
#include "header.cpp"
#endif

void upperenvelope(double *grid_a, int Na, double *m_vec, double *c_vec, double *inv_w_vec, bool use_inv_w,
                   double *grid_m, int Nm,
                   double *c_ast_vec, double *v_ast_vec, double d, par_struct *par)
{
        
    for(int im = 0; im < Nm; im++){
        c_ast_vec[im] = 0;
        v_ast_vec[im] = -HUGE_VAL;
    }

    // constraint
    // the constraint is binding if the common m is smaller
    // than the smallest m implied by EGM step (m_vec[0])

    int im = 0;
    while(im < Nm && grid_m[im] <= m_vec[0]){
            
        // a. consume all
        c_ast_vec[im] = grid_m[im];

        // b. value of choice
        double u = utility::func(c_ast_vec[im],d,par);
        if(use_inv_w){
            v_ast_vec[im] = u + (-1.0/inv_w_vec[0]);
        } else {
            v_ast_vec[im] = u + inv_w_vec[0];
        }
        im += 1;
    
    }

    // upper envellope
    // apply the upper envelope algorithm
    
    for(int ia = 0; ia < Na-1; ia++){

        // a. a inteval and w slope
        double a_low  = grid_a[ia];
        double a_high = grid_a[ia+1];
        
        double inv_w_low  = inv_w_vec[ia];
        double inv_w_high = inv_w_vec[ia+1];

        if(a_low > a_high){continue;}

        double inv_w_slope = (inv_w_high-inv_w_low)/(a_high-a_low);
        
        // b. m inteval and c slope
        double m_low  = m_vec[ia];
        double m_high = m_vec[ia+1];

        double c_low  = c_vec[ia];
        double c_high = c_vec[ia+1];

        double c_slope = (c_high-c_low)/(m_high-m_low);

        // c. loop through common grid
        for(int im = 0; im < Nm; im++){

            // i. current m
            double m = grid_m[im];

            // ii. interpolate?
            bool interp = (m >= m_low) && (m <= m_high);     
            bool extrap_above = ia == Na-2 && m > m_vec[Na-1];

            // iii. interpolation (or extrapolation)
            if(interp | extrap_above){

                // o. implied guess
                double c_guess = c_low + c_slope * (m - m_low);
                double a_guess = m - c_guess;

                // oo. implied post-decision value function
                double inv_w = inv_w_low + inv_w_slope * (a_guess - a_low);               

                // ooo. value-of-choice
                double u = utility::func(c_guess,d,par);
                double v_guess;
                if(use_inv_w){
                    v_guess = u + (-1.0/inv_w);
                } else {
                    v_guess = u + inv_w;
                }

                // oooo. update
                if(v_guess > v_ast_vec[im]){
                    v_ast_vec[im] = v_guess;
                    c_ast_vec[im] = c_guess;
                }
            } // interp / extrap
        } // im
    } // ia

} // upperenvelope

void upperenvelope_2d(double *grid_a, int Na, double *m_vec, double *c_vec, double *inv_w_vec, bool use_inv_w,
                   double *grid_m, int Nm,
                   double *c_ast_vec, double *v_ast_vec, double d1, double d2, par_struct *par)
{
        
    for(int im = 0; im < Nm; im++){
        c_ast_vec[im] = 0;
        v_ast_vec[im] = -HUGE_VAL;
    }

    // constraint
    // the constraint is binding if the common m is smaller
    // than the smallest m implied by EGM step (m_vec[0])

    int im = 0;
    while(im < Nm && grid_m[im] <= m_vec[0]){
            
        // a. consume all
        c_ast_vec[im] = grid_m[im];

        // b. value of choice
        double u = utility::func_2d(c_ast_vec[im],d1,d2,par);
        if(use_inv_w){
            v_ast_vec[im] = u + (-1.0/inv_w_vec[0]);
        } else {
            v_ast_vec[im] = u + inv_w_vec[0];
        }
        im += 1;
    
    }

    // upper envellope
    // apply the upper envelope algorithm
    
    for(int ia = 0; ia < Na-1; ia++){

        // a. a inteval and w slope
        double a_low  = grid_a[ia];
        double a_high = grid_a[ia+1];
        
        double inv_w_low  = inv_w_vec[ia];
        double inv_w_high = inv_w_vec[ia+1];

        if(a_low > a_high){continue;}

        double inv_w_slope = (inv_w_high-inv_w_low)/(a_high-a_low);
        
        // b. m inteval and c slope
        double m_low  = m_vec[ia];
        double m_high = m_vec[ia+1];

        double c_low  = c_vec[ia];
        double c_high = c_vec[ia+1];

        double c_slope = (c_high-c_low)/(m_high-m_low);

        // c. loop through common grid
        for(int im = 0; im < Nm; im++){

            // i. current m
            double m = grid_m[im];

            // ii. interpolate?
            bool interp = (m >= m_low) && (m <= m_high);     
            bool extrap_above = ia == Na-2 && m > m_vec[Na-1];

            // iii. interpolation (or extrapolation)
            if(interp | extrap_above){

                // o. implied guess
                double c_guess = c_low + c_slope * (m - m_low);
                double a_guess = m - c_guess;

                // oo. implied post-decision value function
                double inv_w = inv_w_low + inv_w_slope * (a_guess - a_low);               

                // ooo. value-of-choice
                double u = utility::func_2d(c_guess,d1,d2,par);
                double v_guess;
                if(use_inv_w){
                    v_guess = u + (-1.0/inv_w);
                } else {
                    v_guess = u + inv_w;
                }

                // oooo. update
                if(v_guess > v_ast_vec[im]){
                    v_ast_vec[im] = v_guess;
                    c_ast_vec[im] = c_guess;
                }
            } // interp / extrap
        } // im
    } // ia

} // upperenvelope