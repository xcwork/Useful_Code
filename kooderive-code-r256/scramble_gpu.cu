//
//
//                                                                                             scramble_gpu.cu
//
//



#include "scramble_gpu.h"
#include "cudaMacros.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cutil.h>
#include <cutil_inline.h>


texture<unsigned int, 1> tex_scrambler;


__global__ void scramble_kernel(unsigned int* device_data, int number_dim, int paths)
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;
    
   
    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;
    
    
    int pathsPerThread = 1 + (( paths -1)/width);
    
    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        if (pathNumber < paths)
        {
             unsigned int* data_location = device_data+pathNumber;
             
             for (int d=0; d < number_dim; ++d)
             {
                   unsigned int scrambler = tex1Dfetch(tex_scrambler,d);

                   
                   data_location[d*paths]^= scrambler;
	//			   data_location[d*paths]|=7;
             
             }
                
             
            
            
        }
    }
}





void scramble_gpu( unsigned int* device_data, unsigned int*   scramblers, int number_dim, int paths)
{
   
    const int threadsperblock = 64;
    const int maxBlocks = 65535;
    
    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;
    
    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

      ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);
  
    // Fix the number of threads
    dimBlock.x = threadsperblock;
   
     // Bind the array to the texture
         
     cudaBindTexture(NULL,tex_scrambler,scramblers);
         
        CUT_CHECK_ERR("scramble_gpu execution failed before entering kernel\n");
   
      scramble_kernel<<<dimGrid , dimBlock >>>(device_data,  number_dim,  paths);
    
    
         CUT_CHECK_ERR("scramble_kernel execution failed after entering kernel\n");     
    
         cudaUnbindTexture(tex_scrambler);
   
}
