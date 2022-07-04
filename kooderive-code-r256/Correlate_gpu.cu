// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3


/*
 * correlated variates
 
 The objective is to multiply a large number of paths by the same matrix. 
 
 We should think
 number of paths ~10.000                 N
 number of factors ~ 5                         F            
 number of correlated rates ~ 60  n
 
 The input is a table of (quasi-)random numbers. All paths factor 0 then all paths factor 1 then all paths factor 2 and so on... 
 
 Also need an input matrix of size n \times F. 
 
 The output is a table of log stock or rate increments. All paths rate 0 then all paths rate 1 and so on... 
 
 */

#include "Correlate_gpu.h"
#include "cudaMacros.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cutil.h>

#include <cutil_inline.h>


texture<float, 1, cudaReadModeElementType> tex_correlation;
texture<float, 1, cudaReadModeElementType> tex_A;

texture<int, 1, cudaReadModeElementType> tex_alive;

__device__ void correlation_kernel_texture_device(          int input_dataOffset, // randon numbers, offset for tex_correlation texture
                                                                                                 float* output_data, // correlated rate increments 
                                                                                                 int AOffset,
                                                                                                 int factors, 
                                                                                                 int out_dimensions,
                                                                                                 int number_paths,
                                                                                                 int aliveIndex
)
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;
    
   
    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;
    
    
    int pathsPerThread = 1 + (( number_paths -1)/width);
    
    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        if (pathNumber < number_paths)
        {
            
            int first_variateOffset = input_dataOffset+pathNumber*factors;
            float* first_output = output_data+pathNumber;
            
            int roffset =AOffset+aliveIndex*factors;
            int outoffset =aliveIndex*number_paths;
            
            
            for (int r=aliveIndex; r  < out_dimensions; ++r)            
            {
                float total =0.0;
                
                 for (int f=0; f < factors; ++f)
                    total +=  tex1Dfetch(tex_A,roffset+f)*
                                                                         tex1Dfetch(tex_correlation,first_variateOffset+f);
                                           
                 
                 roffset +=    factors;
                 first_output[outoffset] = total;
                 outoffset+= number_paths;
                             
        }
        
   }
   
}
}


__global__ void correlation_kernel_texture_multistep(   int input_dataOffset, // random numbers, offset for tex_correlation texture
                                                                                                int alive_offset, // offset for tex_alive
                                                                                                 float* output_data, // correlated rate increments 
                                                                                                 int AOffset, // correlator 
                                                                                                 int AoffsetPerStep, // should be the size of A or zero
                                                                                                 int factors, 
                                                                                                 int out_dimensions,
                                                                                                 int number_paths,
                                                                                                 int steps
)
{
     int inputLayerSize =factors*number_paths; 
     int input_data_this_step_offset =input_dataOffset;

        for (int s =0; s < steps; ++s)  
        {
                int aliveIndex=tex1Dfetch(tex_alive,s+alive_offset);
                float* output_data_this_step = output_data + out_dimensions*number_paths*s;
                 int AOffsetThisStep = AOffset + s*AoffsetPerStep;;
                 correlation_kernel_texture_device(input_data_this_step_offset, output_data_this_step, AOffsetThisStep, factors, out_dimensions, number_paths, aliveIndex);     
                 input_data_this_step_offset+=   inputLayerSize;
        }
}


extern "C"
void correlated_paths_multistep_GPU(    float* input_data_device, // randon numbers
                                                                                        float* output_data_device, // correlated rate increments 
                                                                                        float* A_device, 
                                                                                        int* alive_device,
                                                                                                 int A_offset_per_step,
                                                                                                 int factors, 
                                                                                                 int out_dimensions,
                                                                                                 int number_paths,
                                                                                                 int steps)
{

     const int threadsperblock = 64;
    const int maxBlocks = 65535;
    
    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(number_paths-1)/threadsperblock;
    
    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;
  
    // Fix the number of threads
    dimBlock.x = threadsperblock;

   Timer h1;

   CUT_CHECK_ERR("correlated_paths_GPU execution failed before entering kernel\n");
   

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    
        // set texture parameters
         tex_correlation.addressMode[0] = cudaAddressModeWrap;
         tex_correlation.addressMode[1] = cudaAddressModeWrap;
         tex_correlation.filterMode = cudaFilterModeLinear;
         tex_correlation.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_correlation, input_data_device, channelDesc);
 
         tex_A.addressMode[0] = cudaAddressModeWrap;
         tex_A.addressMode[1] = cudaAddressModeWrap;
         tex_A.filterMode = cudaFilterModeLinear;
         tex_A.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_A, A_device, channelDesc);
         
         tex_alive.addressMode[0] = cudaAddressModeWrap;
         tex_alive.addressMode[1] = cudaAddressModeWrap;
         tex_alive.filterMode = cudaFilterModeLinear;
         tex_alive.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_alive, alive_device, channelDesc);
         
         
        CUT_CHECK_ERR("correlated_paths_multistep_GPU textures execution failed before entering kernel\n");
   
        int offset=0;
        
           
        ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);
        correlation_kernel_texture_multistep<<<dimGrid , dimBlock >>>(offset,
                                                                                                offset,
                                                                                                 output_data_device, // correlated rate increments  
                                                                                                 offset, 
                                                                                                 A_offset_per_step,
                                                                                                 factors, 
                                                                                                 out_dimensions,
                                                                                                 number_paths,
                                                                                                 steps);
    
         CUT_CHECK_ERR("correlation_kernel_texture_multistep execution failed after entering kernel\n");     
    
         cudaUnbindTexture(tex_correlation);
  
         cudaUnbindTexture(tex_A);
         
         std::cout << " time taken for correlating numbers: " 
            << h1.timePassed() << std::endl;
         
     

}
