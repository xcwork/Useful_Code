//
//
//                                      pathwise_gold.h
//
//
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3

#ifndef PATHWISE_GOLD_H
#define PATHWISE_GOLD_H
#include <gold/pragmas.h>
#include <vector>
#include <gold/MatrixFacade.h> 
#include <gold/math/typedefs_math_gold.h>
/*
input paths should be paths \times steps
inputNormals should be paths \times steps

we allow multiple vegas in case the product generates cash-flows at varying times,
input the number of steps up to each cash-flow determination time

output is
omega_delta -- paths \time steps entries, derivative of S_{s} wrt S_{s-1} on path p
omega_vegas paths \times steps, derivative of S_{s} wrt \sigma_{s} on path p

*/

 void pathwiseBSweights_gold(const MatrixConstFacade<Realv>& inputPaths,
                              const MatrixConstFacade<Realv>& inputNormals,
                              int normalOffset,
                              const std::vector<Realv>& stepSizeSquareRoots,
                              Realv sigma,
                              Realv S0,
                              int paths,
                              int steps,
                              MatrixFacade<Realv>& w_deltas,
                              MatrixFacade<Realv>& w_vegas,
                              MatrixFacade<Realv>& omega_deltas,
                              MatrixFacade<Realv>& omega_vegas
                              );

#endif

