//
//
//                                                                                                                                          Asian_Test.h
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3

/*

The purpose of this file is automate creation and deletion of cudaStreams
*/

#include <cudaWrappers/cudaStreamWrapper.h>

cudaStreamWrapper::cudaStreamWrapper()
{
    cutilSafeCall(cudaStreamCreate(&inner_Stream));
}


cudaStreamWrapper::~cudaStreamWrapper()
{
    cutilSafeCall(cudaStreamDestroy(inner_Stream));
}


cudaStream_t& cudaStreamWrapper::operator*()
{
    return inner_Stream;
}

cudaError_t cudaStreamWrapper::query()
{
     return cudaStreamQuery(inner_Stream);
}



cudaError_t cudaStreamWrapper::synchronize()
{
   return cudaStreamSynchronize(inner_Stream);
}


