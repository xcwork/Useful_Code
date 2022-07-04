// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3
// note derived from code provided by NVIDIA 


#include "drift_add_main.h"
#include <iostream>
#include <cutil_inline.h>
#include <math.h>
#include <vector>
#include "drift_add_gpu.h"



void drift_add_device(thrust::device_vector<float>&device_fixed_matrix,  // matrix to add
                                                                                                 thrust::device_vector<float>& device_data, // data to add to 
                                                                                                 int steps, 
                                                                                                 int number_rates, 
                                                                                                 int paths,
                                                                                                 bool useTexture)
{
    
     drift_add_gpu(  thrust::raw_pointer_cast(&device_fixed_matrix[0]), // randon numbers
                                                                                                 thrust::raw_pointer_cast(&device_data[0]), // correlated rate increments 
                                                                                                 steps,
                                                                                                 number_rates,
                                                                                                 paths,
                                                                                                 useTexture);
}


