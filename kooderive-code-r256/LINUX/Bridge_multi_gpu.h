// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3



#ifndef MULTI_BRIDGE_GPU_H
#define MULTI_BRIDGE_GPU_H

#include <gold/Bridge_gold.h>

// takes in device pointers
extern "C"
void brownianBridgeMultiGPUReorder(float* input,
                                                                                                 float* output,
                                                                                                int n_poweroftwo,
                                                                                                 int factors,
                                                                                                 int number_paths,
                                                                                                 BrownianBridgeMultiDim<float>::ordering allocator,
                                                                                                 bool useTextures
                                                                                                 );

#endif
