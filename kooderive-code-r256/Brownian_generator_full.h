// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3


//   Brownian_generator_full_h


#ifndef BROWNIAN_GENERATOR_FULL_H
#define BROWNIAN_GENERATOR_FULL_H
#include <gold/pragmas.h> 
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


float BrownianGenerationRoutine(int paths, 
                                                        int pathOffset, 
                                                        int factors, 
                                                        int stepsForEvolution, 
                                                        int powerOfTwoForVariates,
                                                        thrust::device_vector<unsigned int>& scrambler_device, 
// buffers
                                                        thrust::device_vector<unsigned int>& SobolInts_buffer_device, 
                                                        thrust::device_vector<float>& quasiRandoms_buffer_device, 
                                                        thrust::device_vector<float>& bridgeVariates_device
                                                         );


float BrownianGenerationMainRoutine(int paths, 
                                                       int pathOffset, 
                                                       int factors, 
                                                       int stepsForEvolution, 
                                                       int powerOfTwoForVariates,
                                                       const thrust::host_vector<unsigned int>& scrambler_host, 
                                                       thrust::host_vector<float>& bridgeVariates_host,
                                                       float &innerTime
                                                        );

#endif
