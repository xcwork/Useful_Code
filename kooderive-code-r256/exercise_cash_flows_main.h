//                  exercise_cash_flows_main.cu
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3


#ifndef EXERCISE_CASH_FLOWS_MAIN_H
#define EXERCISE_CASH_FLOWS_MAIN_H

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
																) ;

#endif
