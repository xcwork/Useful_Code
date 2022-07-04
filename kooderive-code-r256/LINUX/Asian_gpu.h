// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3



#ifndef ASIAN_GPU_H
#define ASIAN_GPU_H
#include <gold/pragmas.h>
#include <thrust/host_vector.h>

#include <thrust/device_vector.h>

extern "C"
void AsianCallGPU_kernel_dev (thrust::device_vector<float>& input_normals,
                              int normalsDimensions, // must be greater than or equal to stepsPerPath
                              int totalPaths, 
                              int stepsPerPath,
                              const thrust::host_vector<float>& logDrifts_host, 
                              const thrust::host_vector<float>& logSds_host, 
                              float logSpot,
                              float df,
                              float strikeArithmetic,
                              thrust::device_vector<float>& outputDataArithmetic,
                              float strikeGeometric,
                              thrust::device_vector<float>&outputDataGeometric);
#endif
