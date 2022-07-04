//
//
//                                      Correlation_main.cpp
//
//

// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3
// note derived from code provided by NVIDIA 


#include "Correlation_main.h"
#include <iostream>
#include <cutil_inline.h>
#include <math.h>
#include <thrust/version.h>
#include <vector>
#include <thrust/host_vector.h>
#include "Correlate_gpu.h"





void correlated_paths_main_multistep_device(thrust::device_vector<float>& input_data_device, // randon numbers
                                                                                                 thrust::device_vector<float>& output_data_device, // correlated rate increments 
                                                                                                thrust::device_vector<float>& A_device, // correlator 
                                                                                                thrust::device_vector<int>& alive_device,
                                                                                                int A_offsetPerStep,
                                                                                                 int factors, 
                                                                                                 int out_dimensions,
                                                                                                 int number_paths,
                                                                                                 int steps)
{
    
     correlated_paths_multistep_GPU(  thrust::raw_pointer_cast(&input_data_device[0]), // randon numbers
                                                                                                 thrust::raw_pointer_cast(&output_data_device[0]), // correlated rate increments 
                                                                                                thrust::raw_pointer_cast(&A_device[0]), // correlator 
                                                                                                 thrust::raw_pointer_cast(&alive_device[0]), // correlator 
                                                                                                 A_offsetPerStep,
                                                                                                 factors, 
                                                                                                 out_dimensions,
                                                                                                 number_paths,
                                                                                                 steps);

}


