#ifndef MAIN
#define TRANS
#include "header.cpp"
#endif

namespace trans {
    
double p_plus_func(double p, double psi, par_struct* par){
    double p_plus = p*psi;
    p_plus = MAX(p_plus,par->p_min); 
    p_plus = MIN(p_plus,par->p_max); 
    return p_plus;
}

double n_plus_func(double d, par_struct* par){
    double n_plus = (1.0-par->delta)*d;
    n_plus = MIN(n_plus,par->n_max);
    return n_plus;
}

double n1_plus_func(double d1, par_struct* par){
    double n1_plus = (1.0-par->delta1)*d1;
    n1_plus = MIN(n1_plus,par->n_max);
    return n1_plus;
}

double n2_plus_func(double d2, par_struct* par){
    double n2_plus = (1.0-par->delta2)*d2;
    n2_plus = MIN(n2_plus,par->n_max);
    return n2_plus;
}

double m_plus_func(double a, double p_plus, double xi_plus, par_struct* par){
    double y_plus = p_plus*xi_plus;
    double m_plus = par->R*a+ y_plus;
    return m_plus;

}

double x_plus_func(double m_plus, double n_plus, par_struct* par){
    return m_plus + (1.0-par->tau)*n_plus;
}

double x_plus_full_func(double m_plus, double n1_plus, double n2_plus, par_struct* par){
    return m_plus + (1.0-par->tau1)*n1_plus + (1.0-par->tau2)*n2_plus;
}

double x_plus_d1_func(double m_plus, double n1_plus, par_struct* par){
    return m_plus + (1.0-par->tau1)*n1_plus;
}

double x_plus_d2_func(double m_plus, double n2_plus, par_struct* par){
    return m_plus + (1.0-par->tau2)*n2_plus;
}


} // namespace