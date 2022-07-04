//
//                                  cashFlowGen_earlyEx_product_main.h
//
//


// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3

#ifndef CASH_FLOW_GEN_EE_PRODUCT_MAIN_H
#define CASH_FLOW_GEN_EE_PRODUCT_MAIN_H

#include <cashFlowGen_earlyExProduct_gpu.h>
#include <thrust/device_vector.h>

template<class R,class S,class T>
void cashFlowGeneratorEE_main(
                             thrust::device_vector<float>&  genFlows1_device,
                             thrust::device_vector<float>&  genFlows2_device, 
                             thrust::device_vector<float>&  productData_device,
                             thrust::device_vector<float>&  exerciseValueDataFloat_device,
                             thrust::device_vector<int>&   exerciseValueDataInt_device,
                             thrust::device_vector<float>&  exerciseStrategyDataFloat_device,
                             int exerciseStrategyDataFloat_size,
                             thrust::device_vector<int>&  exerciseStrategyDataInt_device,
                             int exerciseStrategyDataInt_size,
                             int paths, 
                             int numberSteps,
                              int numberExerciseDates,
                             thrust::device_vector<bool>&   isExerciseDate_device,
                             thrust::device_vector<float>&  rates1_device, 
                             thrust::device_vector<float>&  rates2_device, 
                             thrust::device_vector<float>&  rates3_device, 
                             thrust::device_vector<float>&  strategyVariables_device,
                             int strategyVariables_maxSize,
                             thrust::device_vector<float>&  forwards_device, 
                             thrust::device_vector<float>&  discRatios_device
							 ,
							 thrust::device_vector<float>& estCValues_device, 
                             thrust::device_vector<int>&  exerciseIndices_Device
							 )   
                             
{
  cashFlowGeneratorEE_gpu<R,S,T>(
                             thrust::raw_pointer_cast(&  genFlows1_device[0]),
                             thrust::raw_pointer_cast(&  genFlows2_device[0]), 
                             thrust::raw_pointer_cast(&  productData_device[0]),
                             thrust::raw_pointer_cast(&  exerciseValueDataFloat_device[0]),
                             thrust::raw_pointer_cast(&exerciseValueDataInt_device[0]),
                             thrust::raw_pointer_cast(&  exerciseStrategyDataFloat_device[0]),
                             exerciseStrategyDataFloat_size,
                             thrust::raw_pointer_cast(&  exerciseStrategyDataInt_device[0]),
                             exerciseStrategyDataInt_size,
                             paths, 
                             numberSteps,
                             numberExerciseDates,
                             thrust::raw_pointer_cast(&isExerciseDate_device[0]),
                             thrust::raw_pointer_cast(&  rates1_device[0]), 
                             thrust::raw_pointer_cast(&  rates2_device[0]), 
                             thrust::raw_pointer_cast(&  rates3_device[0]), 
                             thrust::raw_pointer_cast(&  strategyVariables_device[0]),
                             strategyVariables_maxSize,
                             thrust::raw_pointer_cast(&  forwards_device[0]), 
                             thrust::raw_pointer_cast(&  discRatios_device[0])
							 ,
							  thrust::raw_pointer_cast(&  estCValues_device[0]), 
                              thrust::raw_pointer_cast(&  exerciseIndices_Device[0])
							 );
}


#endif
