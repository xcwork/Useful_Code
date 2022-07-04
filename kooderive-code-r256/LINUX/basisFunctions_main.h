
//
//
//                  basisFunctions_main.h
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3


#ifndef BASIS_FUNCTIONS_MAIN_H
#define BASIS_FUNCTIONS_MAIN_H

#include <thrust/device_vector.h>
                                                                
void exerciseTimeDeterminationUsingLSAQuadraticSimpleBasis_main(thrust::device_vector<int>& stoppingTimes_dev, // output
                                                                  thrust::device_vector<float>& rebates_dev, 
																  thrust::device_vector<int>& LSAIntData_dev, 
																  thrust::device_vector<float>& LSAFloatData_dev, 
															      int paths, 
                                                                  int numberSteps,   
                                                                  thrust::device_vector<float>& rates1_dev,  
                                                                  thrust::device_vector<float>& rates2_dev ,  
                                                                  thrust::device_vector<float>& rates3_dev,
																  thrust::device_vector<float>& forwards_dev, 
                                                                  thrust::device_vector<float>& discountRatios_dev
																) ;
#endif
