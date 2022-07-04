//
//
//                          Max_estimation_gold.h
//
//
// (c) Mark Joshi 2014

#ifndef MAX_ESTIMATION_GOLD_H
#define MAX_ESTIMATION_GOLD_H

#include <gold/MatrixFacade.h>
#include <gold/Mersenne_gold.h>
class MaxEstimatorForAB
{
public:

    MaxEstimatorForAB(unsigned long seed, int numberpoints);

    void GetEstimate(int paths, 
                     const std::vector<double>& means,
                     const std::vector<double>& sds, 
                     const std::vector<bool>& exerciseOccurred,
                     double& biasMean,
                     double& biasSe);


private:
    MersenneTwisterUniformRng rng_; 

    int numberPoints_;
};



#endif
