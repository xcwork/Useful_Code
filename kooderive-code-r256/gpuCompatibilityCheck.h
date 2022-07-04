/*
                                    gpuCompatibilityCheck.h


                                    (c) Mark Joshi 2014
*/


#ifndef GPU_COMPATIBILITY_CHECK
#define GPU_COMPATIBILITY_CHECK
#include <cuda_runtime.h>
// GPU must already be set
class ConfigCheckForGPU
{
public:
    ConfigCheckForGPU(){}

    bool checkConfig(int &threads, 
                                    int& blocks, 
                                    size_t estimatedMem, 
                                    size_t  sharedMemRequirement,  
                                    float& excessMemRatio, 
                                    float& excessSharedRatio) const;
      bool checkConfig( dim3& dimBlock, dim3& dimGrid,
                        
                                    size_t estimatedMem, 
                                    size_t  sharedMemRequirement,  
                                    float& excessMemRatio, 
                                    float& excessSharedRatio) const;

      
   bool  checkConfig( dim3& dimBlock, dim3& dimGrid) const;

   
    bool checkConfig(int &threads, 
                                    int& blocks) const;

    bool checkGlobalMem(size_t estimatedMem, float& excessMemRatio ) const;

private:
};

#endif

