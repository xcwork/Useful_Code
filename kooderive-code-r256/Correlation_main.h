// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3


//   Correlation_main.h


#ifndef CORRELATION_MAIN_H
#define CORRELATION_MAIN_H
#include <gold/pragmas.h> 
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


// takes device arrays and passes to GPU routine, the the  many steps at once kernel

void correlated_paths_main_multistep_device(thrust::device_vector<float>& input_data_device, // randon numbers
                                                                                                 thrust::device_vector<float>& output_data_device, // correlated rate increments 
                                                                                                thrust::device_vector<float>& A_device, // correlator 
                                                                                                thrust::device_vector<int>& alive_device,
                                                                                                int A_offsetPerStep,
                                                                                                 int factors, 
                                                                                                 int out_dimensions,
                                                                                                 int number_paths,
                                                                                                 int steps);
#endif


