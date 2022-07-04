//
//
//                                  Transpose_gpu.cu                                                                                                                       drift_add_gpu.cu
//
//

#include "cudaMacros.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cutil.h>

#include <cutil_inline.h>

#include <CUDAConditionalMacros.h>

texture<float, 1, cudaReadModeElementType> tex_input_matrix;


__global__ void transpose_using_ldg_kernel(const float* const input_data, size_t inputrows, size_t inputcols,
                                           float* output_ptr)
{
    // coalesce output not inputs

    size_t data_size = inputrows*inputcols;

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( data_size -1)/width);
    size_t pathNumber =  bwidth*bx+tx;
   
    for (size_t l=0; l < pathsPerThread; ++l)
    {
         if (pathNumber < data_size)
        {
        
            size_t outc =pathNumber % inputrows;
            size_t outr = pathNumber / inputrows;

            size_t inloc = outc // i.e. in r
                *inputcols
                + outr;

            output_ptr[pathNumber] = LDG(input_data+inloc); 

             pathNumber+= width;

        }
    }

}

double transpose_using_ldg_gpu(const float* const input_data, size_t inputrows, size_t inputcols,
                               float* output_ptr)
{
    int threads= 128;

    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = static_cast<unsigned int>(1+(inputrows*inputcols-1)/threads);


    // Fix the number of threads
    dimBlock.x = threads;
    ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);


    Timer t1;

    transpose_using_ldg_kernel<<<dimGrid , dimBlock >>>( input_data,  inputrows,  inputcols,
        output_ptr);

    cutilSafeCall(cudaThreadSynchronize());
    double time = t1.timePassed();

    return time;

}


__global__ void transpose_using_texture_kernel(size_t inputrows, size_t inputcols,
                                           float* output_ptr)
{
    // coalesce output not inputs

    size_t data_size = inputrows*inputcols;

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( data_size -1)/width);
    size_t pathNumber =  bwidth*bx+tx;
   
    for (size_t l=0; l < pathsPerThread; ++l)
    {
         if (pathNumber < data_size)
        {
        
            size_t outc =pathNumber % inputrows;
            size_t outr = pathNumber / inputrows;

            size_t inloc = outc // i.e. in r
                *inputcols
                + outr;

            output_ptr[pathNumber] = tex1Dfetch(tex_input_matrix,inloc); 

             pathNumber+= width;

        }
    }

}
double transpose_using_texture_gpu(const float* const input_data, size_t inputrows, size_t inputcols,
                               float* output_ptr)
{
    int threads= 1024;

    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = static_cast<unsigned int>(1+(inputrows*inputcols-1)/threads);


    // Fix the number of threads
    dimBlock.x = threads;
    ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);
        
    // set texture parameters
         cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  
    tex_input_matrix.addressMode[0] = cudaAddressModeWrap;
    
    tex_input_matrix.addressMode[1] = cudaAddressModeWrap;
    
    tex_input_matrix.filterMode = cudaFilterModeLinear;
    
    tex_input_matrix.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
    
    cudaBindTexture( NULL, tex_input_matrix, input_data, channelDesc);
         
  

    Timer t1;

    transpose_using_texture_kernel<<<dimGrid , dimBlock >>>(  inputrows,  inputcols,
        output_ptr);

    cutilSafeCall(cudaThreadSynchronize());

    double time = t1.timePassed();

        cudaUnbindTexture(tex_input_matrix);

    return time;

}
