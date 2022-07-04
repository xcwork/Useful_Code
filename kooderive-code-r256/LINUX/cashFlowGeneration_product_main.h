//
//
//                                                                          cashFlowGeneration_product_main.h
//
//
// (c) Mark Joshi 2010,2011
// This code is released under the GNU public licence version 3




#ifndef CASH_FLOW_GENERATION_PRODUCT_MAIN_H
#define  CASH_FLOW_GENERATION_PRODUCT_MAIN_H
#include <gold/pragmas.h> 
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
                                                                 thrust::device_vector<float>& discRatios);
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
                                                                 thrust::device_vector<float>& discRatios);
#endif
