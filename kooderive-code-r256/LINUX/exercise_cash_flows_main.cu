
//
//
//                  exercise_cash_flows_main.cu
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3


#include "exercise_cash_flows_main.h"
#include "exercise_cash_flows_gpu.h"

#include <thrust/device_vector.h>
                                                                
void exercise_cash_flows_gpu_main(thrust::device_vector<float>& genFlows_dev, // output
																  thrust::device_vector<int>& firstIndex_dev, 
																  thrust::device_vector<int>& secondIndex_dev,
																  thrust::device_vector<float>& theta_dev, 
														   	      thrust::device_vector<float>& exercise_values_dev, 
															      thrust::device_vector<int>& firstIndexExercise_dev, 
																  thrust::device_vector<int>& secondIndexExercise_dev,
																  thrust::device_vector<float>& thetaExercise_dev, 
														   	      thrust::device_vector<int>& exerciseTimes_dev,
															      int paths, 
                                                                  int numberSteps
																) 
 {

	 exercise_cash_flows_gpu (thrust::raw_pointer_cast(& genFlows_dev[0]),  // gen flows will be modified
										    thrust::raw_pointer_cast(& firstIndex_dev[0]), // will be changed
										    thrust::raw_pointer_cast(& secondIndex_dev[0]), // will be changed
										    thrust::raw_pointer_cast(& theta_dev[0]),  // will be changed
										    thrust::raw_pointer_cast(& exercise_values_dev[0]),
										    thrust::raw_pointer_cast(& firstIndexExercise_dev[0]),
										    thrust::raw_pointer_cast(& secondIndexExercise_dev[0]),
										   thrust::raw_pointer_cast(&  thetaExercise_dev[0]),
										   thrust::raw_pointer_cast(&  exerciseTimes_dev[0]), 
										   paths, 
										   numberSteps) ;
}
