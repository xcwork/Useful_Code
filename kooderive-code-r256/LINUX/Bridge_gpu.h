// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3

#ifndef BRIDGE_GPU_H
#define BRIDGE_GPU_H



// takes in device pointers
extern "C"
void brownianBridgeGPU_constant_memory(int n_vectors, int n_poweroftwo, float* input,  float* output);


#endif
