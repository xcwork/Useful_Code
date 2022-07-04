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

#ifndef CUDA_EVENT_WRAPPER
#define CUDA_EVENT_WRAPPER

#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#include <utility> 

class cudaEventWrapper
{

public:
    cudaEventWrapper();
    cudaEventWrapper(int flags);

    ~cudaEventWrapper();

    cudaEvent_t& operator*();

    cudaError_t query();
    cudaError_t record();
    cudaError_t record(cudaStream_t stream);
    cudaError_t synchronize();

    std::pair<float, cudaError_t > timeSince(cudaEventWrapper& startEvent);


private:
    cudaEventWrapper(const cudaEventWrapper&){};
    cudaEvent_t inner_Event;


};

#endif //CUDA_EVENT_WRAPPER

