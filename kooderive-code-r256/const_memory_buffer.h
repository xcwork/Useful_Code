//
//
//                                                                          Const_memory_buffer.h
//
//
// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3

#ifndef CONST_MEMORY_BUFFER_H
#define CONST_MEMORY_BUFFER_H

#include <thrust/host_vector.h>
#include <vector>

#define BUFFER_SIZE 1024

 __constant__ __device__ float dev_const_memory_buffer[BUFFER_SIZE]; 





#endif
