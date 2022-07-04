// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3


/*
 * Brownian bridge example
 
 */

#include <gold/Bridge_gold.h>

#include "cudaMacros.h"
/*
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
*/
#include <iostream>
#include <cuda_runtime.h>
#include <cutil.h>

#include <cutil_inline.h>

#include <cudaWrappers/cudaTextureBinder.h>
#include <gpuCompatibilityCheck.h>

#define MAX_BRIDGE_SIZE 256
#define MAX_BRIDGE_POWER_TWO 8

texture<float, 1, cudaReadModeElementType> tex_inputs;

__device__ __constant__ int dev_const_indexTodo[MAX_BRIDGE_SIZE];
__device__ __constant__ int dev_const_indexLeft[MAX_BRIDGE_SIZE];
__device__ __constant__ int dev_const_indexRight[MAX_BRIDGE_SIZE];

__device__ __constant__ int dev_const_variateToUseForThisIndex[MAX_BRIDGE_SIZE];

__device__ __constant__ float dev_const_rightScalars[MAX_BRIDGE_SIZE];
__device__ __constant__ float dev_const_midScalars[MAX_BRIDGE_SIZE];
__device__ __constant__ float dev_const_leftScalars[MAX_BRIDGE_SIZE];


__device__ __constant__ int dev_const_indexRightOneSided[MAX_BRIDGE_POWER_TWO];
__device__ __constant__ int dev_const_indexToDoOneSided[MAX_BRIDGE_POWER_TWO];

__device__ __constant__ float dev_const_rightNotLeftScalars[MAX_BRIDGE_POWER_TWO];
__device__ __constant__ float dev_const_midNotLeftScalars[MAX_BRIDGE_POWER_TWO];



// this one uses the auxiliary data that is already in constant memory
// the objective is to reduce the amount of time spent on non-cached memory access
// here we have the variates arranged so that the first dimension comes first for all paths
// then the second dimension and so on
// so ith variate of path j is located at
// input_data + j + i*number_paths for input
// and at
// output_data + j + i*number_paths
// for output
__global__ void brownianBridgeGPU_const__transposedData_kernel(float firstMidScale,                                                                        
                                                                                                 float* input_data,
                                                                                                 float* output_data,
                                                                                                 int PowerOfTwo,
                                                                                                 int number_dimensions,
                                                                                                 int number_paths
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
            float* input_data_offset =  input_data + pathNumber;
            float* output_data_offset =  output_data + pathNumber;
         
           
             *(output_data_offset+(number_dimensions-1)*number_paths) = input_data_offset[0]*firstMidScale;
   
            for (int i=0; i < PowerOfTwo; ++i)
            {
                int index = dev_const_indexToDoOneSided[i];
                float rscalar= dev_const_rightNotLeftScalars[i];
                float bv = output_data_offset[dev_const_indexRightOneSided[i]*number_paths];
                float variate = input_data_offset[dev_const_variateToUseForThisIndex[index]*number_paths];
                float volscalar = dev_const_midNotLeftScalars[i];
                output_data_offset[index*number_paths] = rscalar* bv+ volscalar*variate;                  
            }

            for (int i=0; i < number_dimensions- PowerOfTwo-1; ++i)
           {
                int index = dev_const_indexTodo[i];
                

                output_data_offset[index*number_paths] =  
                                                                             dev_const_rightScalars[i]*output_data_offset[dev_const_indexRight[i]*number_paths] 
                                                                           + dev_const_leftScalars[i]*output_data_offset[dev_const_indexLeft[i]*number_paths] 
                                                                           + dev_const_midScalars[i]*input_data_offset[dev_const_variateToUseForThisIndex[index]*number_paths];                  
           }
           
           // now do successive differencing
           
            for (int i=1; i < number_dimensions; ++i)
           {                
                int index = number_dimensions-i;
                output_data_offset[index*number_paths] -=     output_data_offset[(index-1)*number_paths];
           }
           

        
        }
   }
}

__global__ void brownianBridgeGPU_const__transposedData_kernel_texture(float firstMidScale,                                                                        
                                                                                                 int input_dat_offset,
                                                                                                 float* output_data,
                                                                                                 int PowerOfTwo,
                                                                                                 int number_dimensions,
                                                                                                 int number_paths
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
            int input_data_offset =  input_dat_offset + pathNumber;
            float* output_data_offset =  output_data + pathNumber;
         
           
             *(output_data_offset+(number_dimensions-1)*number_paths) = tex1Dfetch(tex_inputs, input_data_offset)*firstMidScale;
   
            for (int i=0; i < PowerOfTwo; ++i)
            {
                int index = dev_const_indexToDoOneSided[i];
                float rscalar= dev_const_rightNotLeftScalars[i];
                float bv = output_data_offset[dev_const_indexRightOneSided[i]*number_paths];
                float variate =tex1Dfetch(tex_inputs, input_data_offset+dev_const_variateToUseForThisIndex[index]*number_paths);
                float volscalar = dev_const_midNotLeftScalars[i];
                output_data_offset[index*number_paths] = rscalar* bv+ volscalar*variate;                  
            }

            for (int i=0; i < number_dimensions- PowerOfTwo-1; ++i)
           {
                int index = dev_const_indexTodo[i];
                

                output_data_offset[index*number_paths] =  
                                                                             dev_const_rightScalars[i]*output_data_offset[dev_const_indexRight[i]*number_paths] 
                                                                           + dev_const_leftScalars[i]*output_data_offset[dev_const_indexLeft[i]*number_paths] 
                                                                           + dev_const_midScalars[i]*tex1Dfetch(tex_inputs, input_data_offset+dev_const_variateToUseForThisIndex[index]*number_paths);                  
           }
           
           // now do successive differencing
           
            for (int i=1; i < number_dimensions; ++i)
           {                
                int index = number_dimensions-i;
                output_data_offset[index*number_paths] -=     output_data_offset[(index-1)*number_paths];
           }
           

        
        }
   }
}





extern "C"
void brownianBridgeGPU_constant_memory(int n_vectors, int n_poweroftwo, float* input,float* output)
{
/*
    unsigned int hTimer;
    double       time;
    cutilCheckError(cutCreateTimer(&hTimer));
    cutilCheckError(cutResetTimer(hTimer));
    cutilCheckError(cutStartTimer(hTimer));
*/
    BrownianBridge<float> bb(n_poweroftwo);
    
    int dimensions(bb.dimensions());

    float  firstMidScale(bb.getFirstMidScalar() );
  

    COPYCONSTANTMEMORYFLOAT(dev_const_rightNotLeftScalars,bb.getrightNotLeftScalars());
    COPYCONSTANTMEMORYFLOAT(dev_const_midNotLeftScalars, bb.getmidNotLeftScalars());
  
   COPYCONSTANTMEMORYINT(dev_const_indexRightOneSided, bb.getindexRightOneSided());
   COPYCONSTANTMEMORYINT(dev_const_indexToDoOneSided, bb.getindexToDoOneSided());
  
    COPYCONSTANTMEMORYFLOAT(dev_const_rightScalars, bb.getrightScalars() );
    COPYCONSTANTMEMORYFLOAT(dev_const_midScalars, bb.getmidScalars() );
    COPYCONSTANTMEMORYFLOAT(dev_const_leftScalars, bb.getleftScalars() );

    COPYCONSTANTMEMORYINT(dev_const_indexTodo, bb.getindexTodo());
    COPYCONSTANTMEMORYINT(dev_const_indexLeft, bb.getindexLeft());
    COPYCONSTANTMEMORYINT(dev_const_indexRight, bb.getindexRight());
    
    COPYCONSTANTMEMORYFLOAT(dev_const_variateToUseForThisIndex, bb.getvariateToUseForThisIndex() );
   
    const int threadsperblock = 512;
    const int maxBlocks = 65535;
    
    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(n_vectors-1)/threadsperblock;
    
    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;
  
    // Fix the number of threads
    dimBlock.x = threadsperblock;


     cudaTextureFloatBinder inputsBinder(tex_inputs, input);

    bool useTexture = true;
    
     ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);

    
   CUT_CHECK_ERR("brownianBridgeGPU_kernel execution failed before entering kernel\n");
      
      if (useTexture)
             brownianBridgeGPU_const__transposedData_kernel_texture<<<dimGrid , dimBlock >>>(firstMidScale,                      
                                                                                                 0, // offset for texture
                                                                                                 output,
                                                                                                 n_poweroftwo,
                                                                                                 dimensions,
                                                                                                 n_vectors);
        
      else
      

            // Execute GPU kernel
         brownianBridgeGPU_const__transposedData_kernel<<<dimGrid , dimBlock >>>(firstMidScale,                      
                                                                                                 input,
                                                                                                 output,
                                                                                                 n_poweroftwo,
                                                                                                 dimensions,
                                                                                                 n_vectors);
         

                                                                                                 	
         CUT_CHECK_ERR("brownianBridgeGPU_kernel execution failed\n");
                                                                                               
              

}
