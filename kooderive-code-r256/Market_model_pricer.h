//
//
//                                                                                                                                         Market_model_pricer.h
//
//
// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3

#ifndef  MARKET_MODEL_PRICER_H
#define MARKET_MODEL_PRICER_H
#include <gold/pragmas.h> 
#include <thrust/host_vector.h>
#include <vector>

struct LMMPricerRoutineData
{
    int paths;
    int rates; 
    int factors; 
    int stepsForEvolution; 
    int powerOfTwoForVariates;
    thrust::host_vector<unsigned int> scrambler_host;
    thrust::host_vector<float> pseudoRoots_host;
    thrust::host_vector<float> fixedDrifts_host; 
    thrust::host_vector<float> displacements_host;
    thrust::host_vector<float> initial_rates_host; 
    thrust::host_vector<float> initial_log_rates_host; 
    thrust::host_vector<float> taus_host; 
    thrust::host_vector<float> initial_drifts_host; 
    std::vector<int> aliveIndices; 
    thrust::host_vector<int> alive_host;
    thrust::host_vector<float> auxData;
    std::vector<float> firstCashFlowsTimes; 
    std::vector<float> secondCashFlowsTimes;
    std::vector<float> rateTimes;
    bool useSharedWhereChoice;
    
};

 float LMMPricerRoutine(const LMMPricerRoutineData& inputs,
                                           int scrambleOffset,
                                           int pathOffset, 
                                            int DeviceToUse=-1, // -1 indicates use MaxGflops device
											bool fermiArch=false,
											bool useSharedForDiscounting = false
                                                        );

void LMMPricerRoutineMultiBatch(const LMMPricerRoutineData& inputs,
                                           int pathOffsetPerBatch,
                                           int scrambleOffsetPerBatch,
                                           int baseScrambleOffset,
                                           int basePathOffset, 
                                           int batches,
                                           int DeviceToUse,
                                            float* outputs,
                                            int outputsOffset,
                                            int outputsStep,
                                            bool doDiscounts,
                                            bool newBridge,
											bool fermiArch,
											bool useSharedForDiscounting
                                                        );

#endif
