//
//
//                  LS_basis_main.h
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3



#ifndef LS_BASIS_MAIN_H
#define LS_BASIS_MAIN_H

#include <gold/pragmas.h> 
#include <thrust/device_vector.h>
void adjoinBasisVariablesCaller_main(bool uselog,
									 thrust::device_vector<int>& integerData_device, //  data specific to the basis variables
		                  thrust::device_vector<float>& floatData_device, // data specific to the basis variables
                          thrust::device_vector<float>&  forwards_device,
                          thrust::device_vector<float>&  discountRatios_device,
                          thrust::device_vector<float>&  rate1_device,
                          thrust::device_vector<float>&  rate2_device,
                          thrust::device_vector<float>&  rate3_device,
                          int numberPathsPerBatch,
                          int totalNumberOfPaths, 
                          int numberPathsPreviouslyDone,
                          int numberStepsAndRates, 
                          int numberExerciseTimes,
                          thrust::device_vector<int>&  exerciseIndices_device, // the indices of the exercise times amongst the evolution times
                          thrust::device_vector<int>&  exerciseIndicators_device, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                          thrust::device_vector<float>&  basisFunctionVariables_device // output location
                          );


#endif
