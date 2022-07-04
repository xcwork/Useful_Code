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

#include <cudaWrappers/cudaEventWrapper.h>

cudaEventWrapper::cudaEventWrapper()
{
    cutilSafeCall(cudaEventCreate(&inner_Event));
}

cudaEventWrapper::cudaEventWrapper(int flags)
{
    cutilSafeCall(cudaEventCreate(&inner_Event,flags));
}

cudaEventWrapper::~cudaEventWrapper()
{
    cutilSafeCall(cudaEventDestroy(inner_Event));
}


cudaEvent_t& cudaEventWrapper::operator*()
{
    return inner_Event;
}

cudaError_t cudaEventWrapper::query()
{
     return cudaEventQuery(inner_Event);
}

cudaError_t cudaEventWrapper::record()
{
    return cudaEventRecord(inner_Event);
    
}

cudaError_t cudaEventWrapper::record(cudaStream_t stream)
{
    return cudaEventRecord(inner_Event,stream);
    
}

cudaError_t cudaEventWrapper::synchronize()
{
   return cudaEventSynchronize(inner_Event);
}


std::pair<float, cudaError_t > cudaEventWrapper::timeSince(cudaEventWrapper& startEvent)
{
    float t;
    cudaError_t errVal( cudaEventElapsedTime(&t, startEvent.inner_Event,inner_Event));

    return std::pair<float,cudaError_t>(t,errVal);
}
