
//
//
//                  basisFunctions_main.cu
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3


#include "basisFunctions_main.h"
#include "basisFunctions.h"

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
																) 
 {

	exerciseTimeDeterminationUsingLSAQuadraticSimpleBasis(thrust::raw_pointer_cast(&  stoppingTimes_dev[0]), 
							   thrust::raw_pointer_cast(&   rebates_dev[0]), 
							   thrust::raw_pointer_cast(&  LSAIntData_dev[0]),
							   thrust::raw_pointer_cast(&  LSAFloatData_dev[0]),
							    paths, 
							    numberSteps,                                  
							   thrust::raw_pointer_cast(&   rates1_dev[0]), 
							   thrust::raw_pointer_cast(&   rates2_dev[0]), 
							   thrust::raw_pointer_cast(&   rates3_dev[0]), 
							   thrust::raw_pointer_cast(&   forwards_dev[0]), 
							   thrust::raw_pointer_cast(&   discountRatios_dev[0]));
}
