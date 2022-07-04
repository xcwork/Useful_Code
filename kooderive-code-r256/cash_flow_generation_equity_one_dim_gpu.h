

//
//                                  cash_Flow_Generation_equity_gpu.h
//
//

// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3

/*
Routines are in the header file since they are templatized on the product.

Takes in a path of equity prices
Generates a final pay-off

*/

#ifndef CASH_FLOW_GENERATION_EQUITY_ONE_DIM_H
#define CASH_FLOW_GENERATION_EQUITY_ONE_DIM_H
#include <gpuCompatibilityCheck.h>


texture<float, 1, cudaReadModeElementType> tex_aux_data;

template<class T>
__global__ void cashFlowGeneratorEquityOneDim_kernel(float* genFlows, 
										 int texture_offset, int paths, int numberSteps,
                                                                            float* stockPaths)
{
    int bx = blockIdx.x;
    int tx =  threadIdx.x;
    
   
    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;
    
    int pathsPerThread = 1 + (( paths -1)/width);
    T obj(texture_offset,numberSteps);
    
    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        int dataOffset = pathNumber;

        if (pathNumber < paths)
        {
            float payoff = obj.getPayOff(stockPaths,paths,pathNumber);
    
            genFlows[dataOffset] = payoff;
                            
        }
    }
}

template<class T>
void cashFlowGeneratorEquityOneDim_gpu(float* genFlows, float* aux_data, int paths, int numberSteps,
                                            float* stockPaths)
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
   
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    
        // set texture parameters
    tex_aux_data.addressMode[0] = cudaAddressModeWrap;
    tex_aux_data.addressMode[1] = cudaAddressModeWrap;
    tex_aux_data.filterMode = cudaFilterModeLinear;
    tex_aux_data.normalized = false;    // access with normalized texture coordinates
    cudaBindTexture( NULL, tex_aux_data, aux_data, channelDesc);

    int offset=0;
     ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);

    cashFlowGeneratorEquityOneDim_kernel<T><<<dimGrid , dimBlock >>>(genFlows,offset,paths,numberSteps,stockPaths);
    cudaUnbindTexture(tex_aux_data);
    

}
#endif
