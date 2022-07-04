//
//
//                                                                                                                                                         drift_add_gpu.cu
//
//

// add a fixed matrix to a large number of different matrices

#include "drift_add_gpu.h"
#include "cudaMacros.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cutil.h>

#include <cutil_inline.h>

/*
memory layout:

if i is the number of the path  
stepNumber is the step of the simulation, ie the row of the matrix 

out_dimensions is the number of columns

r is the column, (i.e. which rate in the LMM)


the data to be acted on is structured as 

 i + stepNumber * number_paths*out_dimensions +r*number_paths;

the matrix to be added is  
 stepNumber *out_dimensions +r;
*/

texture<float, 1, cudaReadModeElementType> tex_drift_add;

__global__ void drift_add_kernel(float* device_data, int steps, int number_rates, int paths)
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
             float* data_location = device_data+pathNumber;
             
             for (int s=0; s < steps; ++s)
             {
                int rowstart = s*number_rates;
                for (int r=0; r < number_rates; ++r)
                {
                    float* thisDataPiece = data_location +s*paths* number_rates + r*paths;
                    
                    int texlocation = rowstart+r;
                    float adder = tex1Dfetch(tex_drift_add,texlocation);
                    (*thisDataPiece) += adder;
                }
              }
            
        }
    }
}

__global__ void drift_add_global_memory_kernel(float* adderMatrix, float* device_data, int steps, int number_rates, int paths)
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
             float* data_location = device_data+pathNumber;
             
             for (int s=0; s < steps; ++s)
             {
                int rowstart = s*number_rates;
                for (int r=0; r < number_rates; ++r)
                {
                    float* thisDataPiece = data_location +s*paths* number_rates + r*paths;
                    
                    int texlocation = rowstart+r;
                    float adder =adderMatrix[texlocation];
                    (*thisDataPiece) += adder;
                }
              }
            
        }
    }
}



// note device_fixed_matrix must have been allocated  using cudaMalloc (or some wrapper thereof)
void drift_add_gpu(float* device_fixed_matrix, float* device_data, int steps, int number_rates, int paths, bool useTexture)
{

     
     
   
    const int threadsperblock = 64;
    const int maxBlocks = 65535;
    
    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;
    
    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;
  
    // Fix the number of threads
    dimBlock.x = threadsperblock;
   
            
   ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);
    if (useTexture)
    {
         // allocate array and copy image data
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    
        // set texture parameters
         tex_drift_add.addressMode[0] = cudaAddressModeWrap;
         tex_drift_add.addressMode[1] = cudaAddressModeWrap;
         tex_drift_add.filterMode = cudaFilterModeLinear;
         tex_drift_add.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_drift_add, device_fixed_matrix, channelDesc);
         
         
        CUT_CHECK_ERR("drift_add_gpu execution failed before entering kernel\n");
   
         drift_add_kernel<<<dimGrid , dimBlock >>>(device_data, steps,  number_rates, paths);
    
         CUT_CHECK_ERR("drift_add_gpu execution failed after entering kernel\n");     
    
         cudaUnbindTexture(tex_drift_add);
      }
      else
      {
            drift_add_global_memory_kernel<<<dimGrid , dimBlock >>>(device_fixed_matrix,device_data, steps,  number_rates, paths);
      }
}
