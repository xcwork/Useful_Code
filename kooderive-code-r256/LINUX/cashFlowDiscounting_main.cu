//
//
//                                                                          cashFlowDiscounting_main.cpp
//
//
// (c) Mark Joshi 2010, 2013
// This code is released under the GNU public licence version 3


#include "cashFlowDiscounting_main.h"
#include "cashFlowDiscounting_gpu.h"

                                                                  
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
                                                                  thrust::device_vector<float>& summedDiscountedFlows) // output
 {
      cashFlowDiscounting_gpu( thrust::raw_pointer_cast(&  firstIndex[0]), 
                                                                  thrust::raw_pointer_cast(& secondIndex[0]),
                                                                   thrust::raw_pointer_cast(&  theta[0]), 
                                                                  thrust::raw_pointer_cast(& discountRatios[0]), 
                                                                   thrust::raw_pointer_cast(&  genFlows[0]), 
                                                                   thrust::raw_pointer_cast(& numeraireValues[0]),
                                                                    paths, 
                                                                  numberSteps, 
                                                                  useTextures,
																  useShared,
                                                                   thrust::raw_pointer_cast(&  discountedFlows[0]), // output
                                                                  thrust::raw_pointer_cast(&  summedDiscountedFlows[0])); // output
 }



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
                                      thrust::device_vector<float>& discountedFlows_dev)
{
     cashFlowDiscounting_partial_gpu(genIndex_dev.size(),
                                     thrust::raw_pointer_cast(&genIndex_dev[0]), 
                                     thrust::raw_pointer_cast(&firstIndex_dev[0]), 
                                     thrust::raw_pointer_cast(&secondIndex_dev[0]),
                                     thrust::raw_pointer_cast(&theta_dev[0]), 
                                     thrust::raw_pointer_cast(&targetIndices_dev[0]),
                                     thrust::raw_pointer_cast(&discountRatios_dev[0]), 
                                     thrust::raw_pointer_cast(&genFlows_dev[0]), 
                                     thrust::raw_pointer_cast(&numeraireValues_dev[0]),
                                     paths, 
                                     numberSteps, 
                                     thrust::raw_pointer_cast(&discountedFlows_dev[0]));
}
