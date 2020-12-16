//////////////////////////
// 1. external includes //
//////////////////////////

// standard C++ libraries

#include <windows.h>
#include <stdio.h>
#include <cmath>
#include <omp.h>

///////////////
// 2. macros //
///////////////

#define MAX(X,Y) ((X)>(Y)?(X):(Y))
#define MIN(X,Y) ((X)<(Y)?(X):(Y))
#define BOUND(X,A,B) MIN(MAX(X,A),B)

#define EPS 1e-8
#define EXPORT extern "C" __declspec(dllexport)

/////////////////
// 3. includes //
/////////////////

// a. generic
#include "nlopt-2.4.2-dll64\nlopt.h"
#include "HighResTimer_class.hpp"
#include "logs.cpp"
#include "index.cpp"

#ifndef LINEAR_INTERP
#include "linear_interp.cpp"
#endif

// b. structs
#include "par_struct.cpp"
#include "sol_struct.cpp"
#include "sim_struct.cpp"

typedef struct {
    
    int t;
    double p, n, m, x;
    double *inv_w, *inv_v_keep;
    par_struct *par;
    sol_struct *sol;
    double n1, n2;
    int i_p, i_n, i_x;

} solver_struct;

// c. local modules
#ifndef GOLDEN_SECTION_SEARCH
#include "golden_section_search.cpp"
#endif

#ifndef UTILITY
#include "utility.cpp"
#endif

#ifndef TRANS
#include "trans.cpp"
#endif

#ifndef POST_DECISION
#include "post_decision.cpp"
#endif

#ifndef UPPERENVELOPE
#include "upperenvelope.cpp"
#endif