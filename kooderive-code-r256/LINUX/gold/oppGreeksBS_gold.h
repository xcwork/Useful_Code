
//
//                              oppGreeksBS_gold.h
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3


#ifndef OPP_DELTAS_VEGAS_BS_GOLD
#define OPP_DELTAS_VEGAS_BS_GOLD


#include <vector>
#include <gold/pragmas.h>
#include <gold/MatrixFacade.h> 
#include <gold/math/typedefs_math_gold.h>

//    compute omega and w weights for deltas and vegas of log processes wrt
//    initial log S_0 and sttdev per step
//
//
void oppDeltasVegasBS_gold(const MatrixConstFacade<Realv>& inputLogPaths,
                           const MatrixConstFacade<Realv>& inputPaths,
                      const MatrixConstFacade<Realv>& inputUniforms, // the uniforms used to produced the paths
                      const MatrixConstFacade<Realv>& inputGaussians, // the Gaussians used to produced the paths
                      int normalOffset,
                      const std::vector<Realv>& sigmas,
                      const std::vector<Realv>& dTs,
                      const std::vector<Realv>& rootdTs,
                      const std::vector<Realv>& logDrifts,
                      const std::vector<Realv>& stdevs, // should be sigma root T for step 
                      Realv logS0,
                      Realv S0,
                      const std::vector<Realv>& discontinuityLogLevels, // must be increasing
                      int paths,
                      int steps,
                      MatrixFacade<Realv>& deltas_omegas,
                      MatrixFacade<Realv>& deltas_w,
                      MatrixFacade<Realv>& vegas_omegas,
                      MatrixFacade<Realv>& vegas_w
                              );

#endif

