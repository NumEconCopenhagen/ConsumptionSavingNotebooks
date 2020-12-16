#ifndef MAIN
#define UTILITY
#include "header.cpp"
#endif

namespace utility {

double func(double c, double d, par_struct* par)
{
    double dtot = d+par->d_ubar;
    double c_total = pow(c,par->alpha)*pow(dtot,1.0-par->alpha);
    return pow(c_total,1.0-par->rho)/(1.0-par->rho);
}

double func_2d(double c, double d1, double d2, par_struct* par)
{
    double d1tot = d1+par->d1_ubar;
    double d2tot = d2+par->d2_ubar;
    double c_total = pow(c,par->alpha)*pow(d1tot,par->gamma*(1.0-par->alpha))*pow(d2tot,(1-par->gamma)*(1.0-par->alpha));
    return pow(c_total,1.0-par->rho)/(1.0-par->rho);
}

double marg_func(double c, double d, par_struct* par)
{
    double dtot = d+par->d_ubar;
    double c_power = par->alpha*(1.0-par->rho)-1.0;
    double d_power = (1.0-par->alpha)*(1.0-par->rho);
    return par->alpha*pow(c,c_power)*pow(dtot,d_power);
}

double marg_func_2d(double c, double d1, double d2, par_struct* par)
{
    double d1tot = d1+par->d1_ubar;
    double d2tot = d2+par->d2_ubar;
    double c_power = par->alpha*(1.0-par->rho)-1.0;
    double d1_power = par->gamma*(1.0-par->alpha)*(1.0-par->rho);
    double d2_power = (1.0-par->gamma)*(1.0-par->alpha)*(1.0-par->rho);
    return par->alpha*pow(c,c_power)*pow(d1tot,d1_power)*pow(d2tot,d2_power);
}

double inv_marg_func(double q, double d, par_struct* par)
{
    double dtot = d+par->d_ubar;
    double c_power = par->alpha*(1.0-par->rho)-1.0;
    double d_power = (1.0-par->alpha)*(1.0-par->rho);
    double denom = par->alpha*pow(dtot,d_power);
    return pow(q/denom,1.0/c_power);
}

double inv_marg_func_2d(double q, double d1, double d2, par_struct* par)
{
    double d1tot = d1+par->d1_ubar;
    double d2tot = d2+par->d2_ubar;
    double c_power = par->alpha*(1.0-par->rho)-1.0;
    double d1_power = par->gamma*(1.0-par->alpha)*(1.0-par->rho);
    double d2_power = (1.0-par->gamma)*(1.0-par->alpha)*(1.0-par->rho);
    double denom = par->alpha*pow(d1tot,d1_power)*pow(d2tot,d2_power);
    return pow(q/denom,1.0/c_power);
}

} // namespace