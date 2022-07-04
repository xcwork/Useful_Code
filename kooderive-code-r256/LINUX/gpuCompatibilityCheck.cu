/*
gpuCompatibilityCheck.cu


(c) Mark Joshi 2014
*/


#include <gpuCompatibilityCheck.h>
#include <cuda.h>

#include <cutil.h>

#include <cutil_inline.h>
#include <cuda_runtime.h>
   
bool ConfigCheckForGPU::checkConfig(int &threads, 
                                    int& blocks) const
{
 bool result = false;

    int device;
    cudaDeviceProp deviceproperty;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceproperty, device);          

    int deviceThreads = deviceproperty.maxThreadsPerBlock;
    if (deviceThreads < threads)
    {
        result = true;
        threads = deviceThreads;
    }

    int deviceBlocks = deviceproperty.maxGridSize[0];
    if (blocks < deviceBlocks)
    {
        result = true;
        blocks = deviceBlocks;
    }

    return result;
}
bool ConfigCheckForGPU::checkConfig(int &threads, 
                                    int& blocks, 
                                    size_t estimatedMem, 
                                    size_t  sharedMemRequirement,  
                                    float& excessMemRatio, 
                                    float& excessSharedRatio) const
{
    bool result = false;

    int device;
    cudaDeviceProp deviceproperty;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceproperty, device);          

    int deviceThreads = deviceproperty.maxThreadsPerBlock;
    if (deviceThreads < threads)
    {
        result = true;
        threads = deviceThreads;
    }

    int deviceBlocks = deviceproperty.maxGridSize[0];
    if (blocks < deviceBlocks)
    {
        result = true;
        blocks = deviceBlocks;
    }

    size_t sharedMem = deviceproperty.sharedMemPerBlock;
    size_t globalMem = deviceproperty.totalGlobalMem;

    excessMemRatio = static_cast<float>(estimatedMem) /globalMem;

    if (estimatedMem <= globalMem)
        excessMemRatio = 0.0f;

    excessSharedRatio = static_cast<float>(sharedMemRequirement)/sharedMem;

    if (sharedMemRequirement <= sharedMem)
        excessSharedRatio = 0.0f;

    return result;

}


bool ConfigCheckForGPU::checkConfig( dim3& dimBlock, dim3& dimGrid) const
{
    bool result = false;

    int device;
    cudaDeviceProp deviceproperty;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceproperty, device);          
    int* maxthreadsPtr = deviceproperty.maxThreadsDim;


    if (dimBlock.x > static_cast<unsigned int>(maxthreadsPtr[0]))
    {
        dimBlock.x = maxthreadsPtr[0];
    }

    if (dimBlock.y > static_cast<unsigned int>(maxthreadsPtr[1]))
    {
        dimBlock.y = maxthreadsPtr[1];
    }

    if (dimBlock.z > static_cast<unsigned int>(maxthreadsPtr[2]))
    {
        dimBlock.z = maxthreadsPtr[2];
    }

    int deviceThreads = deviceproperty.maxThreadsPerBlock;
    while (static_cast<unsigned int>(deviceThreads) < dimBlock.x*dimBlock.y*dimBlock.y)
    {
        result = true;
        if (dimBlock.x >1)
            --dimBlock.x;
        else
            if  (dimBlock.y >1)
                --dimBlock.y;
            else
                --dimBlock.z;
    }

    int* maxGrids = deviceproperty.maxGridSize;
    if (dimGrid.x > static_cast<unsigned int>(maxGrids[0]))
    {
        dimGrid.x = maxGrids[0];
        result = true;
    }

    if (dimGrid.y > static_cast<unsigned int>(maxGrids[1]))
    {
        dimGrid.y = maxGrids[1];
        result = true;
    }

    if (dimGrid.z > static_cast<unsigned int>(maxGrids[2]))
    {
        dimGrid.z = maxGrids[2];
        result = true;
    }

    return result;
}
bool  ConfigCheckForGPU::checkConfig( dim3& dimBlock, dim3& dimGrid,
                                     size_t estimatedMem, 
                                     size_t  sharedMemRequirement,  
                                     float& excessMemRatio, 
                                     float& excessSharedRatio) const
{
    bool result = false;

    int device;
    cudaDeviceProp deviceproperty;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceproperty, device);          
    int* maxthreadsPtr = deviceproperty.maxThreadsDim;


    if (dimBlock.x > static_cast<unsigned int>(maxthreadsPtr[0]))
    {
        dimBlock.x = maxthreadsPtr[0];
    }

    if (dimBlock.y > static_cast<unsigned int>(maxthreadsPtr[1]))
    {
        dimBlock.y = maxthreadsPtr[1];
    }

    if (dimBlock.z > static_cast<unsigned int>(maxthreadsPtr[2]))
    {
        dimBlock.z = maxthreadsPtr[2];
    }

    int deviceThreads = deviceproperty.maxThreadsPerBlock;
    while (static_cast<unsigned int>(deviceThreads) < dimBlock.x*dimBlock.y*dimBlock.y)
    {
        result = true;
        if (dimBlock.x >1)
            --dimBlock.x;
        else
            if  (dimBlock.y >1)
                --dimBlock.y;
            else
                --dimBlock.z;
    }

    int* maxGrids = deviceproperty.maxGridSize;
    if (dimGrid.x > static_cast<unsigned int>(maxGrids[0]))
    {
        dimGrid.x = maxGrids[0];
        result = true;
    }

    if (dimGrid.y > static_cast<unsigned int>(maxGrids[1]))
    {
        dimGrid.y = maxGrids[1];
        result = true;
    }

    if (dimGrid.z > static_cast<unsigned int>(maxGrids[2]))
    {
        dimGrid.z = maxGrids[2];
        result = true;
    }


    size_t sharedMem = deviceproperty.sharedMemPerBlock;
    size_t globalMem = deviceproperty.totalGlobalMem;

    excessMemRatio = static_cast<float>(estimatedMem) /globalMem;

    if (estimatedMem <= globalMem)
        excessMemRatio = 0.0f;

    excessSharedRatio = static_cast<float>(sharedMemRequirement)/sharedMem;

    if (sharedMemRequirement <= sharedMem)
        excessSharedRatio = 0.0f;

    return result;

}

bool ConfigCheckForGPU::checkGlobalMem(size_t estimatedMem, float& excessMemRatio ) const
{
    bool result = false;

    int device;
    cudaDeviceProp deviceproperty;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceproperty, device);          

      size_t globalMem = deviceproperty.totalGlobalMem;

    excessMemRatio = static_cast<float>(estimatedMem) /globalMem;

    std::cout << " excessMemRatio " << excessMemRatio << "\n";

    if (estimatedMem <= globalMem)
        excessMemRatio = 0.0f;

   
    return result;

}
