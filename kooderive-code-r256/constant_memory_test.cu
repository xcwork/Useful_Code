

#include "constant_memory_test.h"

#include "cudaMacros.h"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cutil.h>
#include <cudaMacros.h>
#include <cutil_inline.h>

namespace const_memory_test_cu
{
    #include "const_memory_accessor.h"
}

__global__ void addConstant(float * data, int constantOffset, int dataSize)
{
    int bx = blockIdx.x;
    int tx =  threadIdx.x;
    
   
    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;
    
    
    int pathsPerThread = 1 + (( dataSize -1)/width);
    
    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        if (pathNumber < dataSize)
        {
            data[pathNumber] += const_memory_test_cu::dev_const_memory_buffer[constantOffset+pathNumber];
      
                             
        }
        
   }
   
}




int testConstantMemory(bool verbose)
{
    int failures=0;
    
    {
    std::vector<float> v(4);
    v[0] = 1.0;
    v[1] = 3.0;
    v[2] = 5.0;
    v[3] = 7.0;
    
    std::vector<float> w;
    
    const_memory_test_cu::SetConstMemory( v, 0);
    const_memory_test_cu::GetConstMemoryContents( w);
    
    for (int i=0; i < v.size(); ++i)
        if (v[i] != w[i])
            {
                    std::cout << "Const memory test failure, " << i << "," << v[i] << "," << w[i] << "\n";
                    return 1;
            }
      
         const int threadsperblock = 64;
    const int maxBlocks = 65535;
    
    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = static_cast<unsigned int>(1+(v.size()-1)/threadsperblock);
    
    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;
  
    // Fix the number of threads
    dimBlock.x = threadsperblock;
      
      std::vector<float> data(4);
      data[0] = 10.0f;
      data[1] = 11.0f;
      data[2] = 12.0f;
      data[3] = 13.0f;
      
      std::vector<float> dataCopy(data);
      
      int offset=0;
      
     addConstant<<<dimGrid , dimBlock >>>(&data[0],offset,static_cast<int>(data.size()));
     
     float tolerance = 1e-5f;
     

     
     for (int i=0; i < data.size(); ++i)
        if (fabs(data[i] - (dataCopy[i]+v[i])) > tolerance)
        {
            std::cout << i << data[i] << v[i] << dataCopy[i] << "\n";
            ++failures; 
      }
      
      
          cudaThreadExit();
      }
    return failures > 0 ? 1 : 0;
}
