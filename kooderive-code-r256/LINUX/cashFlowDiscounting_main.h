//
//
//                                                                          cashFlowDiscounting_main.h
//
//

// (c) Mark Joshi 2010,2011,2013
// This code is released under the GNU public licence version 3

#ifndef cashFlowDiscounting_main_h_123344342
#define cashFlowDiscounting_main_h_123344342
#include <gold/pragmas.h> 
#include <thrust/device_vector.h>
#include <vector>

void cashFlowDiscounting_gpu_main(thrust::device_vector<int>& firstIndex, 
                                                                  thrust::device_vector<int>& secondIndex,
                                                                  thrust::device_vector<float>& theta, 
                                                                  thrust::device_vector<float>& discountRatios, 
                                                                  thrust::device_vector<float>& genFlows, 
                                                                  thrust::device_vector<float>& numeraireValues,
                                                                  int paths, 
                                                                  int numberSteps, 
                                                                  bool useTextures,
																  bool useShared,
                                                                  thrust::device_vector<float>& discountedFlows, // output
                                                                  thrust::device_vector<float>& summedDiscountedFlows); // output



void cashFlowDiscounting_partial_gpu_main(thrust::device_vector<int>& genIndex_dev,
                                          thrust::device_vector<int>& firstIndex_dev, 
                                      thrust::device_vector<int>& secondIndex_dev,
                                      thrust::device_vector<float>& theta_dev, 
                                      thrust::device_vector<int>& targetIndices_dev,
                                      thrust::device_vector<float>& discountRatios_dev, 
                                      thrust::device_vector<float>& genFlows_dev, 
                                      thrust::device_vector<float>& numeraireValues_dev,
                                      int paths, 
                                      int numberSteps, 
                                      thrust::device_vector<float>& discountedFlows_dev);

#endif
