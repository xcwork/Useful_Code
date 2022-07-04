// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3


//   scramble_main.h


#ifndef SCRAMBLE_MAIN_H
#define SCRAMBLE_MAIN_H
#include <gold/pragmas.h> 
#include <vector>

#include <thrust/device_vector.h>


// takes device arrays and passes to GPU routine,

void scramble_main(thrust::device_vector<unsigned int>& input_data_device, // random numbers
                                                                                                thrust::device_vector<unsigned int>& scramblers_device,
                                                                                                 int dimensions,
                                                                                                 int number_paths);
#endif


