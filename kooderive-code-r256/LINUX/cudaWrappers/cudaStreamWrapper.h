//
//
//                                                                                                                                          Asian_Test.h
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3

/*

The purpose of this file is automate creation and deletion of cudaEvents
*/

#ifndef CUDA_STREAM_WRAPPER
#define CUDA_STREAM_WRAPPER

#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <utility> 

class cudaStreamWrapper
{

public:
    cudaStreamWrapper();

    ~cudaStreamWrapper();

    cudaStream_t& operator*();

    cudaError_t query();

    cudaError_t synchronize();


private:
    cudaStreamWrapper(const cudaStreamWrapper&){};
    cudaStream_t inner_Stream;


};

#endif //CUDA_STREAM_WRAPPER

