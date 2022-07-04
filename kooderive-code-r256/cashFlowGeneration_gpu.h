

//
//                                  cashFlowGeneration_gpu.h
//
//

// (c) Mark Joshi 2010, 2014
// This code is released under the GNU public licence version 3

/*
Routines are in the header file since they are templatized on the product.
*/

#ifndef CASH_FLOW_GENERATION_H
#define CASH_FLOW_GENERATION_H
#include <gpuCompatibilityCheck.h>
texture<float, 1, cudaReadModeElementType> tex_aux_data;



template<class T>
__global__ void cashFlowGenerator_kernel(float* genFlows1, 
										 float* genFlows2, 
										 int texture_offset, 
                                         int paths, 
                                         int numberSteps,                                   
                                         float* rates1, 
                                         float* rates2, 
                                         float* rates3, 
                                         float* forwards, 
                                         float* discRatios)
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
        bool done = false;
        if (pathNumber < paths)
        {
    
               obj.newPath();
               
               for (int i =0; i < numberSteps; ++i)
               {
                    float flow1=0.0;
                    float flow2=0.0;
                    if (!done)
                    {               

                        float rate1 = rates1[dataOffset];
    
                        float rate2= rates2[dataOffset];
                        float rate3= rates3[dataOffset];
             
                        done = obj.getCashFlows(flow1,flow2,rate1,rate2,rate3,forwards,discRatios,paths);
                   
                     }
                     genFlows1[dataOffset] = flow1;
                     genFlows2[dataOffset] = flow2;                
                     dataOffset += paths;
            }               
        }
    }
}

template<class T>
void cashFlowGenerator(float* genFlows1, float* genFlows2, float* aux_data, int paths, int numberSteps,
                                            float* rates1, float* rates2, float* rates3, float* forwards, float* discRatios)
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
    cashFlowGenerator_kernel<T><<<dimGrid , dimBlock >>>(genFlows1,genFlows2,offset,paths,numberSteps,rates1,rates2,rates3,forwards,discRatios);
    cudaUnbindTexture(tex_aux_data);
    

}
#endif
