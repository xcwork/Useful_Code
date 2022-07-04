//
//
//                                                                                                                                          Asian_Test.h
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3

/*

The purpose of this file is automate binding and unbinding of textures
*/


#include <cudaWrappers/cudaTextureBinder.h>



cudaTextureFloatBinder::cudaTextureFloatBinder(texture<float, 1, cudaReadModeElementType>& texture_reference, thrust::device_ptr< float > dataPtr)
:
texture_reference_(texture_reference)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    // set texture parameters
    texture_reference_.addressMode[0] = cudaAddressModeWrap;
    texture_reference_.addressMode[1] = cudaAddressModeWrap;
    texture_reference_.filterMode = cudaFilterModeLinear;
    texture_reference_.normalized = false;    // access with normalized texture coordinates
    cudaBindTexture( NULL, texture_reference_, thrust::raw_pointer_cast(dataPtr), channelDesc);

    std::cout << " device ptr float used\n";

}


cudaTextureFloatBinder::~cudaTextureFloatBinder()
{
    cudaUnbindTexture(texture_reference_);
}

cudaTextureIntBinder::cudaTextureIntBinder(texture<int, 1>& texture_reference, 
                                           int* dataPtr)
                                           :
texture_reference_(texture_reference)
{

    cudaBindTexture( NULL, texture_reference_, dataPtr);

}

cudaTextureIntBinder::cudaTextureIntBinder(texture<int, 1>& texture_reference, thrust::device_ptr< int > dataPtr)
:
texture_reference_(texture_reference)
{
    cudaBindTexture( NULL, texture_reference_, thrust::raw_pointer_cast(dataPtr));
      std::cout << " device ptr int used\n";
}

cudaTextureIntBinder::~cudaTextureIntBinder()
{
    cudaUnbindTexture(texture_reference_);
}
