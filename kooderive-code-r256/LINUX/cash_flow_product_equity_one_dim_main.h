
//
//                                  cash_flow_product_equity_one_dim_main.h
//
//


// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3

#ifndef CASH_FLOW_PRODUCT_EQUITY_ONE_DIM_MAIN_H
#define CASH_FLOW_PRODUCT_EQUITY_ONE_DIM_MAIN_H

#include <thrust/device_vector.h>


void payoffGeneratorCallerRatchetPut_main(thrust::device_vector<float>& genFlows,  
                                          thrust::device_vector<float>& aux_data, 
                                          int paths, 
                                          int numberSteps,
                                          thrust::device_vector<float>& stockPaths
                                          );

#endif
