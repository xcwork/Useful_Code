//
//                              likelihoodRatio_gold.h
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3




#ifndef LIKELIHOOD_RATIO_GOLD_H
#define LIKELIHOOD_RATIO_GOLD_H
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
LR_delta -- paths entries
LR_vegas paths \times stepNumbersForVegas.size()

*/

// cf Glasserman 2004, page 404
void likelihoodRatiosBS_gold (const MatrixConstFacade<Realv>& inputPaths,
                              const MatrixConstFacade<Realv>& inputNormals,
                              int normalOffset,
                              const std::vector<Realv>& stepSizeSquareRoots,
                              Realv sigma,
                              Realv S0,
                              int paths,
                              int steps,
                              const std::vector<int>& stepNumbersForVegas,
                              bool cumulativeWeights,
                              std::vector<Realv>& LR_deltas,
                              std::vector<Realv>& LR_vegas
                              );
                                   

#endif
