//
//
//                                     scramble_main.cpp
//
//

// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3
 


#include "scramble_main.h"
#include <iostream>
#include <cutil_inline.h>
#include <math.h>
#include <thrust/version.h>
#include <vector>
#include <thrust/host_vector.h>
#include "scramble_gpu.h"

void scramble_main(thrust::device_vector<unsigned int>& input_data_device, // random numbers
                                                                                                thrust::device_vector<unsigned int>& scramblers_device,
                                                                                                 int dimensions,
                                                                                                 int number_paths)
{
    
 scramble_gpu(  thrust::raw_pointer_cast(&input_data_device[0]), thrust::raw_pointer_cast(&scramblers_device[0]),  dimensions, number_paths);

}


