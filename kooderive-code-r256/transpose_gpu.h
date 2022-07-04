//
//
//                                  Transpose_gpu.h                                                                                                                      drift_add_gpu.cu
//
//

#ifndef TRANSPOSE_GPU_H
#define TRANSPOSE_GPU_H


double transpose_using_ldg_gpu(const float* const input_data, size_t inputrows, size_t inputcols,
                               float* output_ptr);

double transpose_using_texture_gpu(const float* const input_data, size_t inputrows, size_t inputcols,
                               float* output_ptr);
#endif
