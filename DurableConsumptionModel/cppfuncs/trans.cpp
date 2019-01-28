#ifndef MAIN
#define TRANS
#include "header.cpp"
#endif

namespace trans {
    
double p_plus_func(double p, double psi, par_struct *par){
    double p_plus = p*psi;
    p_plus = MAX(p_plus,par->p_min); 
    p_plus = MIN(p_plus,par->p_max); 
    return p_plus;
}

double n_plus_func(double d, par_struct *par){
    double n_plus = (1.0-par->delta)*d;
    n_plus = MIN(n_plus,par->n_max);
    return n_plus;
}

double m_plus_func(double a, double p_plus, double xi_plus, par_struct *par){
    double y_plus = p_plus*xi_plus;
    double m_plus = par->R*a+ y_plus;
    return m_plus;

}

double x_plus_func(double m_plus, double n_plus,par_struct *par){
    return m_plus + (1.0-par->tau)*n_plus;
}

} // namespace