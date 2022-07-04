
//
//                              oppGreeksBS_gold.cpp
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3


#include <gold/oppGreeksBS_gold.h> 

#include <gold/math/Normals_gold.h>
#include <gold/pragmas.h>
#include <gold/MatrixFacade.h> 
#include <algorithm>

//    compute omega and w weights for deltas of log process wrt
//    initial log S_0
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
                              )
{
    int numberStrata = discontinuityLogLevels.size()+1;
    

    for (int p=0; p < paths; ++p)
    {
        Realv y = logS0;
        Realv expy = S0;

        for (int s=0; s < steps; ++s)
        {
            // x is log stock at start of step, y is log stock at end of step
            Realv x =y;
            y= inputLogPaths(p,s);

            Realv expx =expy;
            expy= inputPaths(p,s);



            Realv alphas, alphaDashs, alphaDashVega;
            Realv alphas1, alphaDashs1, alphaDashVega1;

            int stratum = std::upper_bound(discontinuityLogLevels.begin(), discontinuityLogLevels.end(),y)-discontinuityLogLevels.begin();
            int pstratum = stratum-1;


            if (stratum ==numberStrata-1) // on RHS
            {
                alphas1=1;
                alphaDashs1=0;
                alphaDashVega1 =0;
            }
            else
            {   

                Realv cutPoint = (discontinuityLogLevels[stratum] - x- logDrifts[s])/stdevs[s];
                alphas1 = static_cast<Realv>(cumulativeNormal(cutPoint));
                Realv densityVal = static_cast<Realv>(normalDensity(cutPoint));
                alphaDashs1 = -densityVal/stdevs[s];
                alphaDashVega1 = (rootdTs[s]- cutPoint/sigmas[s])*densityVal;

            }

            if (stratum ==0) // on LHS
            {
                alphas=0;
                alphaDashs=0;
                alphaDashVega=0;
            }
            else
            {   
                Realv cutPoint = (discontinuityLogLevels[pstratum] - x- logDrifts[s])/stdevs[s];
                alphas = static_cast<Realv>(cumulativeNormal(cutPoint));
                Realv densityVal = static_cast<Realv>(normalDensity(cutPoint));
                alphaDashs = -densityVal/stdevs[s];
                alphaDashVega =(rootdTs[s]- cutPoint/sigmas[s])*densityVal;
            }

            Realv v = inputUniforms(s,p+normalOffset);
            Realv theta = (v- alphas)/(alphas1-alphas);

            Realv d = (alphaDashs1-alphaDashs)/(alphas1 -alphas);

            Realv gammaV = alphaDashs + theta*(alphaDashs1-alphaDashs);

            deltas_w(p,s) = d/expx;

            Realv icndV =static_cast<Realv>(inverseCumulativeNormalDerivative(v)); 

            Realv omegaDelta = (expy/expx)*(1 + stdevs[s]*icndV*gammaV);

            deltas_omegas(p,s) =omegaDelta;

            Realv dVega =  (alphaDashVega1-alphaDashVega)/(alphas1 -alphas);

            Realv gammaVegaV = alphaDashVega+theta*(alphaDashVega1-alphaDashVega);

            Realv w1= expy*(inputGaussians(s,p+normalOffset)*rootdTs[s]-sigmas[s]*dTs[s]);
            Realv w2 =expy*stdevs[s]*icndV;



            vegas_w(p,s) = dVega;

            vegas_omegas(p,s) = w1 + w2*gammaVegaV;




        }
    
    }

}

