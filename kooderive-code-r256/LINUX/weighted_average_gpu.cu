
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3



/*
weighted basket example

*/
#include "weighted_average_gpu.h"
#include <cuda_runtime.h>
#include <cutil.h>
#include "cudaMacros.h"
#include <cutil_inline.h>
#include <cudaWrappers/cudaTextureBinder.h>


texture<float, 1, cudaReadModeElementType> tex_weights;




__global__ void basketWeightings_kernel(float* inputPaths, float* outputPaths,
                                           int paths,
                                           int numberStocks,
                                           int numberSteps)
                                                                           
{
    int bx = blockIdx.x;
    int tx =  threadIdx.x;
    
    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;
    
    int pathsPerThread = 1 + (( paths -1)/width);
    int layerSize = paths*numberStocks;

    
    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        
        if (pathNumber < paths)
        {
     
                   
            for (int i =0; i < numberSteps; ++i)
            {
               float total=0.0;

               for (int j=0; j < numberStocks; ++j)
               {
                   float weight = tex1Dfetch(tex_weights,j);
                   total += weight*inputPaths[pathNumber + j*paths+i*layerSize];
               }

               outputPaths[pathNumber+i*paths]=total;

                
            }
               
        }
    }
}


                                                                  

extern"C"
void  basketWeightings_gpu(float* inputPaths, float* outputPaths, float* weights,
                                           int paths,
                                           int numberOfStocks,
                                           int numberSteps)
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
   
	cudaTextureFloatBinder texBinder(tex_weights, weights);
	
      ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);
    basketWeightings_kernel<<<dimGrid , dimBlock >>>(inputPaths,  outputPaths,
                                           paths,
                                           numberOfStocks,
                                           numberSteps);

}

