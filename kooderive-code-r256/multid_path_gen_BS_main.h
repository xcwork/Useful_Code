//
//
//                                      MultiD_Path_gen_BS_main.h
//
//
// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3



#ifndef MULTID_PATH_GEN_BS_MAIN_H
#define MULTID_PATH_GEN_BS_MAIN_H


#include <gold/pragmas.h> 
#include <thrust/device_vector.h>

void multi_dim_BS_path_generator_device(thrust::device_vector<float>& logRates_device,
                                                                                           thrust::device_vector<float>& correlatedBrownianIncrements_device,
                                                                                           thrust::device_vector<float>& outputPaths_device, 
                                                                                           int paths,
                                                                                           int rates,
                                                                                           int steps);
#endif
