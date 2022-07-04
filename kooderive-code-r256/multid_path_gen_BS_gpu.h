//
//
//                                                          Multi D Path generator BS GPU
//
//
// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3

#ifndef MULTID_PATH_GEN_BS_GPU_H
#define MULTID_PATH_GEN_BS_GPU_H


extern "C"
void multi_dim_BS_path_generator_kernel( float* logRates_device,
                                                                                           float* correlatedBrownianIncrements_device,
                                                                                           float* outputPaths_device, 
                                                                                           int paths,
                                                                                           int rates,
                                                                                           int steps);

#endif
