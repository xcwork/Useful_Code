
//
//                    cashFlowAggregation_gpu.cu
//
//
//

// (c) Mark Joshi 2011,2014
// This code is released under the GNU public licence version 3



#include <cudaWrappers/cudaTextureBinder.h>
#include <gpuCompatibilityCheck.h>

texture<int, 1> tex_precedingIndex;





__global__ void AggregateFlows_kernel(float* aggregatedFlows,
                                      int totalPaths,
                                      int numberExerciseDates, 
                                      float* genFlows,
                                      int batchPaths, 
                                      int offSetForOutput,
                                      int numberSteps,
									  int start)
{
    int bx = blockIdx.x;
    int tx =  threadIdx.x;
    
    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;
    
    int pathsPerThread = 1 + (( batchPaths -1)/width);

    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
         
        if (pathNumber < batchPaths)
        {
            for (int i=start; i < numberSteps; ++i)
            {
                int j = tex1Dfetch(tex_precedingIndex,i);
                aggregatedFlows[j*totalPaths+pathNumber+offSetForOutput] +=  genFlows[i*batchPaths+pathNumber];
            }
        }
    }
}

                                                             

extern"C"
void AggregateFlows_gpu(
                            float* aggregatedFlows,// output added to not overwritten
                                      int totalPaths,
                                      int numberExerciseDates, 
                                      float* genFlows,
                                      int batchPaths, 
                                      int offSetForOutput,
                                      int numberSteps, 
                                      int* precedingIndex // to texture
									  , int start
                                      ) // output
{
    

    const int threadsperblock = 64;
    const int maxBlocks = 65535;
    
    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(batchPaths-1)/threadsperblock;
    
    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;
  
    // Fix the number of threads
    dimBlock.x = threadsperblock;
   


	cudaTextureIntBinder precedingBinder(tex_precedingIndex,precedingIndex);

    ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);

    AggregateFlows_kernel<<<dimGrid , dimBlock >>>(aggregatedFlows,
                                      totalPaths,
                                      numberExerciseDates, 
                                      genFlows,
                                      batchPaths, 
                                      offSetForOutput,
                                      numberSteps,
									  start);

}
