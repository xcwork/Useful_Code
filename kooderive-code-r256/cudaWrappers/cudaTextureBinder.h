//
//
//
//			cudaTextureBinder.h
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3

#ifndef CUDA_TEXTURE_BINDER_H
#define CUDA_TEXTURE_BINDER_H

#include <thrust/device_ptr.h>

/*
Binds texture, unbinds at end of scope. These are very lightweight objects and so cannot be copied or assigned. 

For floats: parameters are 
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

// set texture parameters
texture_reference_.addressMode[0] = cudaAddressModeWrap;
texture_reference_.addressMode[1] = cudaAddressModeWrap;
texture_reference_.filterMode = cudaFilterModeLinear;
texture_reference_.normalized = false;    // access with normalized texture coordinates
cudaBindTexture( NULL, texture_reference_, dataPtr, channelDesc);

A different constructor could be added if a different configuration was desired. 

*/

class cudaTextureFloatBinder
{
public:
    cudaTextureFloatBinder(texture<float, 1, cudaReadModeElementType>& texture_reference, 
        float* dataPtr)
        :
    texture_reference_(texture_reference)
    {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

        // set texture parameters
        texture_reference_.addressMode[0] = cudaAddressModeWrap;
        texture_reference_.addressMode[1] = cudaAddressModeWrap;
        texture_reference_.filterMode = cudaFilterModeLinear;
        texture_reference_.normalized = false;    // access with normalized texture coordinates
        cudaBindTexture( NULL, texture_reference_, dataPtr, channelDesc);

    }
    cudaTextureFloatBinder(texture<float, 1, cudaReadModeElementType>& texture_reference, thrust::device_ptr< float > dataPtr);
    ~cudaTextureFloatBinder();

private:
    texture<float, 1, cudaReadModeElementType>& texture_reference_;

    cudaTextureFloatBinder(const cudaTextureFloatBinder& orig): texture_reference_(orig.texture_reference_){}
    cudaTextureFloatBinder& operator=(const cudaTextureFloatBinder&){return *this;}

};

class cudaTextureIntBinder
{
public:
    cudaTextureIntBinder(texture<int, 1>& texture_reference, int* dataPtr);
    cudaTextureIntBinder(texture<int, 1>& texture_reference, thrust::device_ptr< int > dataPtr);

    ~cudaTextureIntBinder();

private:
    texture<int, 1>& texture_reference_;

    cudaTextureIntBinder(const cudaTextureIntBinder& orig): texture_reference_(orig.texture_reference_){}
    cudaTextureIntBinder& operator=(const cudaTextureIntBinder&){return *this;}

};

#endif
