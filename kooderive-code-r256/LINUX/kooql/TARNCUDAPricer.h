//
//                                               TARNCUDAPricer.h
//
//
//

#ifndef TARN_CUDA_PRICER_H
#define TARN_CUDA_PRICER_H

#include <vector> 
class MarketModelEvolverLMMPC_CUDA;
void TARNPricerRoutine(const MarketModelEvolverLMMPC_CUDA& evolver,
                                            int batches,
                                            const std::vector<float>& auxData,
                                            const std::vector<float>& firstCashFlowsTimes, 
                                            const std::vector<float>& secondCashFlowsTimes,
                                            std::vector<float>& batchValues_vec
                        );

void TARNPricerRoutineMultiThread(const MarketModelEvolverLMMPC_CUDA& evolver,
                                            int batches,
                                            const std::vector<float>& auxData,
                                            const std::vector<float>& firstCashFlowsTimes, 
                                            const std::vector<float>& secondCashFlowsTimes,
                                            int numberOfThreadsToUse, 
                                            std::vector<float>& batchValues_vec,
                                            bool multiBatch,
                                            bool useShared,
                                            bool mergeDiscounts,
                                            bool newBridge,
                                            std::vector<int> deviceIndex,
											bool fermiArch,
											bool sharedForDiscounting
                        );

#endif
