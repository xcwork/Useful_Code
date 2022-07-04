//
//
//                                      MultiD_Path_gen_BS_main.cpp
//
//
// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3

#include "multid_path_gen_BS_main.h"
#include <iostream>
#include <cutil_inline.h>
#include <math.h>
#include <thrust/version.h>
#include <vector>

#include "multid_path_gen_BS_gpu.h"





void multi_dim_BS_path_generator_device(thrust::device_vector<float>& logRates_device,
                                                                                           thrust::device_vector<float>& correlatedBrownianIncrements_device,
                                                                                           thrust::device_vector<float>& outputPaths_device, 
                                                                                           int paths,
                                                                                           int rates,
                                                                                           int steps)
{

    outputPaths_device.resize(paths*rates*steps);

    multi_dim_BS_path_generator_kernel(   thrust::raw_pointer_cast(&logRates_device[0]),
                                                                                           thrust::raw_pointer_cast(&correlatedBrownianIncrements_device[0]),
                                                                                           thrust::raw_pointer_cast(&outputPaths_device[0]), 
                                                                                           paths,
                                                                                           rates,
                                                                                           steps);
    
}


