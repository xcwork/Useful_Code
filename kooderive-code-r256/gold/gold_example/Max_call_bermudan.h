//
//											max_call_bermudan.h
//
//

// (c) Mark Joshi 2015
// released under GPL v 3.0
#ifndef MAX_CALL_BERMUDAN_GOLD23937483
#define MAX_CALL_BERMUDAN_GOLD23937483

#include<vector>

int Max_call_bermudan();
int Max_call_bermudan(int paths,
    int pathsPerSecondPassBatch,
    int numberSecondPassBatches ,
    int duplicate,
    int numberStocks,
    int factors,
    int stepsForEvolution, 
    int powerOfTwoForVariates,
    double sigma,
    double rho,
    double T,
    double r,
    double d,
    double S0,
    double strike,
    // regresion parameters
    bool useCrossTerms,
    bool normalise ,
    int numberOfExtraRegressions,
    int lowerPathCutoff ,
    double lowerFrac ,
    double upperFrac,
    double initialSDguess ,
    int outerPaths , // parameters for outer sim for upper bound
    int upperSeed ,
    const std::vector<int >& subsimPaths_vec,     // parameters for inner sim for upper bound
    int minPathsForUpper,
    std::vector<int> pathsForGaussianEstimation_vec,  // parameters for MC bias estimation
    int biasseed,
    const std::vector<int>& extraPaths_vec,
    int powerToUse ,
    bool dumpBiases, 
    bool dump
    );
#endif
