//////////////////////////
// 1. external includes //
//////////////////////////

// Standard C++ libraries
#include <windows.h>
#include<iostream>
#define _USE_MATH_DEFINES
#include <cmath>
 
// Tasmanian Sparse Grids
#include "TASMANIAN-7.0/include/TasmanianSparseGrid.hpp"

///////////////
// 2. macros //
///////////////

#define MAX(X,Y) ((X)>(Y)?(X):(Y))
#define MIN(X,Y) ((X)<(Y)?(X):(Y))
#define BOUND(X,A,B) MIN(MAX(X,A),B)

#define EXPORT extern "C" __declspec(dllexport)

/////////////
// 3. SASG //
/////////////


EXPORT void* create(int dim_inputs, int dim_outputs, double tol, int depth_ini, double *lb, double *ub)
{

    auto grid = new TasGrid::TasmanianSparseGrid;
    grid->makeLocalPolynomialGrid(dim_inputs,dim_outputs,depth_ini,1,TasGrid::rule_localp);
    grid->setDomainTransform(lb,ub);

    return grid;

}

EXPORT int getNumNeeded(void* grid_, double tol, bool setSurplusRefinement)
{

    TasGrid::TasmanianSparseGrid* grid = (TasGrid::TasmanianSparseGrid*) grid_;

    // a. set refinement criterion
    if(setSurplusRefinement){
        grid->setSurplusRefinement(tol,TasGrid::refine_fds);
    }

    // b. get number of new points
    return grid->getNumNeeded();

}

EXPORT void getNeededPoints(void *grid_, double *points)
{

    TasGrid::TasmanianSparseGrid* grid = (TasGrid::TasmanianSparseGrid*) grid_;
    grid->getNeededPoints(points);

}

EXPORT void loadNeededPoints(void* grid_, double *values)
{

    TasGrid::TasmanianSparseGrid* grid = (TasGrid::TasmanianSparseGrid*) grid_;
    grid->loadNeededPoints(values);

}

EXPORT void destroy(void* grid_)
{

    TasGrid::TasmanianSparseGrid* grid = (TasGrid::TasmanianSparseGrid*) grid_;
    delete grid;

}

EXPORT void evaluate(void* grid_, double* x, double *y)
{

    TasGrid::TasmanianSparseGrid* grid = (TasGrid::TasmanianSparseGrid*) grid_;
    grid->evaluate(x,y);

}