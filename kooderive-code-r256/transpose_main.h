//
//
//                                  Transpose_main.h                                                                                                                      drift_add_gpu.cu
//
// (c) Mark Joshi 2014

#ifndef TRANSPOSE_MAIN_H
#define TRANSPOSE_MAIN_H

#include <thrust/device_vector.h>

double transpose_using_gpu_main(thrust::device_vector<float>& input_data_dev, size_t inputrows, size_t inputcols,
                               thrust::device_vector<float>& output_dev, bool useTextures = false);

#endif
