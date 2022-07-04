// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3


#include "multid_path_gen_BS_gpu.h"
#include "cudaMacros.h"
#include <cuda_runtime.h>
#include <cutil.h>
#include <cutil_inline.h>
#include <iostream>

texture<float, 1, cudaReadModeElementType> tex_initiallogRates;

__global__ void  multi_dim_BS_path_generator_kernel(int logRates_offset,
                                                                                           float* correlatedBrownianIncrements,
                                                                                           float* outputPaths, 
                                                                                           int paths,
                                                                                           int rates,
                                                                                           int steps)
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
             float* out_data_location = outputPaths+pathNumber;
             float* brownian_location = correlatedBrownianIncrements+pathNumber;
             
             for (int r=0; r < rates; ++r)
             {
                    float x  =    tex1Dfetch(tex_initiallogRates,logRates_offset+r);
                   
                    for (int s=0; s < steps; ++s)
                    {
                            int offset = s*paths* rates + r*paths;
           
                            float* thisDataOutLocation =  out_data_location +offset;
                            float* thisBrownianIncrement = brownian_location + offset;
                            
                            x+= *thisBrownianIncrement;
                            *thisDataOutLocation = exp(x);
                    
                 
                }
              }
            
        }
    }
}




extern "C"
void multi_dim_BS_path_generator_kernel( float* logRates_device,
                                                                                           float* correlatedBrownianIncrements_device,
                                                                                           float* outputPaths_device, 
                                                                                           int paths,
                                                                                           int rates,
                                                                                           int steps)
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
    Timer t1;
    CUT_CHECK_ERR("correlated_paths_GPU execution failed before entering kernel\n");
   
 
               // allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    
        // set texture parameters
         tex_initiallogRates.addressMode[0] = cudaAddressModeWrap;
         tex_initiallogRates.addressMode[1] = cudaAddressModeWrap;
         tex_initiallogRates.filterMode = cudaFilterModeLinear;
         tex_initiallogRates.normalized = false;    // access with normalized texture coordinates

          // Bind the array to the texture
         cudaBindTexture( NULL, tex_initiallogRates, logRates_device, channelDesc);
         
         
        CUT_CHECK_ERR(" multi_dim_BS_path_generator_kernel execution failed before entering kernel\n");
   
        int offset=0;
   
        multi_dim_BS_path_generator_kernel<<<dimGrid , dimBlock >>>(offset, 
                                                                                           correlatedBrownianIncrements_device,
                                                                                           outputPaths_device, 
                                                                                           paths,
                                                                                           rates,
                                                                                           steps);
    
         CUT_CHECK_ERR("multi_dim_BS_path_generator_kernel execution failed after entering kernel\n");     
    
         cudaUnbindTexture(tex_initiallogRates);
  
         
          std::cout << " time taken for path generation: " 
            << t1.timePassed() << std::endl;
         
    
}
