// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3


//   correlate_drift_main.h


#ifndef CORRELATION_DRIFT_MAIN_H
#define CORRELATION_DRIFT_MAIN_H


#include <gold/pragmas.h> 
#include <thrust/device_vector.h>

void correlated_drift_paths_device(    thrust::device_vector<float>& input_data_device, // randon numbers
                                                                                       thrust::device_vector<float>&  output_data_device, // correlated rate increments 
                                                                                       thrust::device_vector<float>&  A_device, 
                                                                                       thrust::device_vector<int>&  alive_device,
                                                                                       thrust::device_vector<float>&  drift_fixed_device, 
                                                                                                 int A_offset_per_step,
                                                                                                 int factors, 
                                                                                                 int out_dimensions,
                                                                                                 int number_paths,
                                                                                                 int steps);
#endif
