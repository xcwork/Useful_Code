//
//
//                                                                          cashFlowGeneration_product_main.cu
//
//
// (c) Mark Joshi 2010,2011
// This code is released under the GNU public licence version 3


#include "cashFlowGeneration_product_main.h"
#include "cashFlowGeneration_product_gpu.h"
#include <thrust/device_vector.h>

extern "C"
void cashFlowGeneratorCallerTARN_main(  thrust::device_vector<float>& genFlows1, 
                                      thrust::device_vector<float>& genFlows2, 
                                      thrust::device_vector<float>& aux_data, 
                                      int paths, 
                                      int numberSteps,
                                      thrust::device_vector<float>& rates1, 
                                      thrust::device_vector<float>& rates2, 
                                      thrust::device_vector<float>& rates3, 
                                      thrust::device_vector<float>& forwards, 
                                      thrust::device_vector<float>& discRatios)
{
    cashFlowGeneratorCallerTARN( thrust::raw_pointer_cast(& genFlows1[0]), 
        thrust::raw_pointer_cast(& genFlows2[0]), 
        thrust::raw_pointer_cast(&  aux_data[0]), 
        paths, 
        numberSteps,
        thrust::raw_pointer_cast(&  rates1[0]), 
        thrust::raw_pointer_cast(&  rates2[0]), 
        thrust::raw_pointer_cast(& rates3[0]), 
        thrust::raw_pointer_cast(&  forwards[0]), 
        thrust::raw_pointer_cast(&  discRatios[0]));
}

extern "C"
void cashFlowGeneratorCallerSwap_main(  thrust::device_vector<float>& genFlows1, 
                                      thrust::device_vector<float>& genFlows2, 
                                      thrust::device_vector<float>& aux_data, 
                                      int paths, 
                                      int numberSteps,
                                      thrust::device_vector<float>& rates1, 
                                      thrust::device_vector<float>& rates2, 
                                      thrust::device_vector<float>& rates3, 
                                      thrust::device_vector<float>& forwards, 
                                      thrust::device_vector<float>& discRatios)
{
    cashFlowGeneratorCallerSwap( thrust::raw_pointer_cast(& genFlows1[0]), 
        thrust::raw_pointer_cast(& genFlows2[0]), 
        thrust::raw_pointer_cast(&  aux_data[0]), 
        paths, 
        numberSteps,
        thrust::raw_pointer_cast(&  rates1[0]), 
        thrust::raw_pointer_cast(&  rates2[0]), 
        thrust::raw_pointer_cast(& rates3[0]), 
        thrust::raw_pointer_cast(&  forwards[0]), 
        thrust::raw_pointer_cast(&  discRatios[0]));
}                                         
