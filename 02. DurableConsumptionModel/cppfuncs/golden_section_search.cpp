#ifndef MAIN
#define GOLDEN_SECTION_SEARCH
#include "header.cpp"
#endif

double golden_section_search(double a, double b, double tol, void *solver_data, double (*obj)(double,void*))
{
        
    double inv_phi = (sqrt(5) - 1) / 2; // 1/phi                                                                                                                
    double inv_phi_sq = (3 - sqrt(5)) / 2; // 1/phi^2                                                                                                                

    // a. distance
    double dist = b - a;
    if(dist <= tol){ 
        return (a+b)/2;
    }

    // b. number of iterations
    int n = ceil(log(tol/dist)/log(inv_phi));

    // c. potential new mid-points
    double c = a + inv_phi_sq * dist;
    double d = a + inv_phi * dist;
    double yc = obj(c,solver_data);
    double yd = obj(d,solver_data);

    // d. loop
    for(int i = 0; i < n-1; i++){
        if(yc < yd){
            b = d;
            d = c;
            yd = yc;
            dist = inv_phi*dist;
            c = a + inv_phi_sq * dist;
            yc = obj(c,solver_data);
        } else {
            a = c;
            c = d;
            yc = yd;
            dist = inv_phi*dist;
            d = a + inv_phi * dist;
            yd = obj(d,solver_data);
        }
    }

    // e. return
    if(yc < yd){
        return (a+d)/2;
    } else {
        return (c+b)/2;
    }

} // function