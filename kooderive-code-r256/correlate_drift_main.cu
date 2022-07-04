//
//
//                                      correlate_drift_main.cpp
//
//

// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3
// note derived from code provided by NVIDIA 


#include "correlate_drift_main.h"

#include "Correlate_drift_gpu.h"

void correlated_drift_paths_device(    thrust::device_vector<float>& input_data_device, // randon numbers
                                                                                       thrust::device_vector<float>&  output_data_device, // correlated rate increments 
                                                                                       thrust::device_vector<float>&  A_device, 
                                                                                       thrust::device_vector<int>&  alive_device,
                                                                                       thrust::device_vector<float>&  drift_fixed_device, 
                                                                                                 int A_offset_per_step,
                                                                                                 int factors, 
                                                                                                 int out_dimensions,
                                                                                                 int number_paths,
                                                                                                 int steps)
{
    correlated_drift_paths_GPU(    thrust::raw_pointer_cast(& input_data_device[0]), // randon numbers
                                                                                        thrust::raw_pointer_cast(&output_data_device[0]), // correlated rate increments 
                                                                                        thrust::raw_pointer_cast(& A_device[0]), 
                                                                                        thrust::raw_pointer_cast(& alive_device[0]),
                                                                                        thrust::raw_pointer_cast(& drift_fixed_device[0]), 
                                                                                                 A_offset_per_step,
                                                                                                factors, 
                                                                                                out_dimensions,
                                                                                                number_paths,
                                                                                                steps);
}


