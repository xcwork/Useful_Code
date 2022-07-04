//
//
//                          mixedLRPathwise_gold.h
//          
//
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3

#ifndef MIXED_LR_PATHWISE_GOLD_H
#define MIXED_LR_PATHWISE_GOLD_H
#include <gold/MatrixFacade.h>
#include <vector>
#include <gold/pragmas.h>
#include <gold/math/typedefs_math_gold.h>
/*

/*
discountedCashFlows is of dimension p \times max number of flows per path
discountedCashFlowStateDerivatives is the same

cashFlowGenerationIndices is paths \times max number of flows per path
It indicates the indices of the steps at which the cashFlows are determined. 

terminationSteps is size paths. It says where to terminate the path.

deltas_omegas is paths \times steps, it indicates the weight to be placed on the pathwise derivative for the next steo
deltas_w is paths \times steps, it indicates the weight on the cash-flow so is the LR part of the derivative
pathwiseGreekEstimates should be of size paths and is the output vector
*/

void mixedLRPathwiseDeltas_gold(const MatrixConstFacade<Realv>& discountedCashFlows,
                                const MatrixConstFacade<Realv>& discountedCashFlowStateDerivatives,
                                const MatrixConstFacade<int>& cashFlowGenerationIndices,
                                const std::vector<int>& terminationSteps,
                                int paths, 
                                const MatrixConstFacade<Realv>& deltas_omegas,
                                const MatrixConstFacade<Realv>& deltas_w,
                                std::vector<Realv>& pathDeltaEstimates  
                              );

void mixedLRPathwiseDeltasVega_gold(const MatrixConstFacade<Realv>& discountedCashFlows,
                                    const MatrixConstFacade<Realv>& discountedCashFlowStateDerivatives,
                                    const MatrixConstFacade<int>& cashFlowGenerationIndices,
                                    const std::vector<int>& terminationSteps,
                                    int maxSteps,
                                    int paths, 
                                    const MatrixConstFacade<Realv>& deltas_omegas,
                                    const MatrixConstFacade<Realv>& deltas_w,
                                    const MatrixConstFacade<Realv>& vegas_omegas,
                                    const MatrixConstFacade<Realv>& vegas_w,                                  
                                    std::vector<Realv>& pathDeltaEstimates , 
                                    std::vector<Realv>& pathVegaEstimates  
                              );

#endif
