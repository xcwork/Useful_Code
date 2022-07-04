//
//
//                  LS_basis_main.cu
//
//
// (c) Mark Joshi 2011,2013
// This code is released under the GNU public licence version 3

#include <LS_Basis_main.h>
#include <LS_Basis_gpu.h>
void adjoinBasisVariablesCaller_main(bool uselog,
									 thrust::device_vector<int>& integerData_device, // to access texture containing data specific to the basis variables
		                  thrust::device_vector<float>& floatData_device, // to access texture containing data specific to the basis variables
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
                          )
{

adjoinBasisVariablesCaller_gpu(uselog,
							   thrust::raw_pointer_cast( &integerData_device[0]), // to access texture containing data specific to the basis variables
		                   thrust::raw_pointer_cast( &floatData_device[0]), // to access texture containing data specific to the basis variables
                           thrust::raw_pointer_cast( &forwards_device[0]),
                           thrust::raw_pointer_cast( &discountRatios_device[0]),
                           thrust::raw_pointer_cast( &rate1_device[0]),
                           thrust::raw_pointer_cast( &rate2_device[0]),
                           thrust::raw_pointer_cast( &rate3_device[0]),
                           numberPathsPerBatch,
                           totalNumberOfPaths, 
                           numberPathsPreviouslyDone,
                           numberStepsAndRates, 
                           numberExerciseTimes,
                           thrust::raw_pointer_cast( &exerciseIndices_device[0]), // the indices of the exercise times amongst the evolution times
                           thrust::raw_pointer_cast( &exerciseIndicators_device[0]), // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                           thrust::raw_pointer_cast( & basisFunctionVariables_device[0]) // output location
                          );
}