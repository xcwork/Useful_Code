// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3



#ifndef BRIDGE_H
#define BRIDGE_H
#include <gold/pragmas.h> 
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "Bridge_gpu.h"
#include "Bridge_multi_gpu.h"


// Device indicates that the routine operates on data already on the device
// and outputs it there
int bridgeMainDevice(int n_vectors, 
                                int n_poweroftwo,
                                thrust::device_vector<float>& dev_inputGPU, 
                                thrust::device_vector<float>& dev_outputGPU,
                                bool report = false);

int MultiDBridgeReordering(int n_vectors, 
                                int n_poweroftwo,
                                int factors,
                                thrust::device_vector<float>& dev_inputGPU, 
                                thrust::device_vector<float>& dev_outputGPU,
                                BrownianBridgeMultiDim<float>::ordering allocator,
                                bool useTextures);

int MultiDBridge(int n_vectors, 
                                int n_poweroftwo,
                                int factors,
                                thrust::device_vector<float>& dev_inputGPU, 
                                thrust::device_vector<float>& dev_outputGPU,
                                BrownianBridgeMultiDim<float>::ordering allocator,
                                bool useTextures,
                                bool report=false);

#endif
