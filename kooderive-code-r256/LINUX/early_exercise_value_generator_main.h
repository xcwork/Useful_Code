//
//
//                          early_exercise_value_generator_main.h
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3

#ifndef EARLY_EXERCISE_VALUE_GENERATOR_MAIN_H
#define EARLY_EXERCISE_VALUE_GENERATOR_MAIN_H

#include <thrust/device_vector.h>

extern "C"
void adjoinExerciseValues_Bermudan_swaption_main(thrust::device_vector<float>& exerciseValues_device,  // output, one per exercise time per path
                               thrust::device_vector<float>& aux_float_data_device, // any float auxiliary data for the exercise value object 
                               thrust::device_vector<int>& aux_int_data_device, //  any int auxiliary data for the exercise value object 
                               thrust::device_vector<int>& exerciseIndices_device, // the indices of the exercise times amongst evolution times
                               thrust::device_vector<int>& exerciseIndicators_device, // boolean indicators of exercise amongst evolution times 
                               int pathsForOutput, // typically totalPaths
                               int outputOffset,
                               int batchPaths, 
                               int numberStepsAndRates, // assumed to be equal 
                               thrust::device_vector<float>& rates1_device, 
                               thrust::device_vector<float>& rates2_device, 
                               thrust::device_vector<float>& rates3_device, 
                               thrust::device_vector<float>& forwards_device, 
                               thrust::device_vector<float>& discRatios_device);

#endif
