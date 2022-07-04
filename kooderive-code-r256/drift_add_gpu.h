// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3

#ifndef DRIFT_ADD_GPU_H
#define DRIFT_ADD_GPU_H

extern "C"
void drift_add_gpu(float* device_fixed_matrix, // must have been allocated on device using CUDAmalloc for memory alignment
                                          float* device_data, 
                                          int steps, 
                                          int number_rates, 
                                          int paths,
                                          bool useTexture);


#endif
