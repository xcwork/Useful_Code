
// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3



/*
* Asian option example

*/
#include "Asian_gpu.h"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cutil.h>
#include "cudaMacros.h"
#include <cutil_inline.h>
#include <gpuCompatibilityCheck.h>
#define MAX_PATH_SIZE 256

__device__ __constant__ float dev_const_drifts[MAX_PATH_SIZE];
__device__ __constant__ float dev_const_logSds[MAX_PATH_SIZE];



// uses transposed variates, so all dimenion zeros before first dimension one and so on 
__global__ //__device__ 
void AsianCallGPU_transposed_kernel(                                                             
    float* input_normals,
    int totalPaths, 
    int stepsPerPath,
    float logSpot0,
    float df,
    float strikeArithmetic,
    float* outputDataArithmetic,
    float strikeGeometric,
    float* outputDataGeometric
    )
{


    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( totalPaths -1)/width);

    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        if (pathNumber < totalPaths)
        {
            float* input_data_offset =  input_normals + pathNumber;  

            float runningSum  = 0.0;
            float runningGeomSum = 0.0;

            float logSpot = logSpot0;

            for (int i=0; i < stepsPerPath; ++i)
            {
                logSpot  +=    input_data_offset[i*totalPaths]*dev_const_logSds[i]+dev_const_drifts[i];
                runningSum += exp(logSpot);
                runningGeomSum += logSpot;
            }

            float payOff = runningSum/stepsPerPath - strikeArithmetic;

            payOff = payOff >0 ? payOff  : 0.0f;

            *(outputDataArithmetic + pathNumber) = payOff*df;

            float gAverage = exp(runningGeomSum/stepsPerPath);

            float gPayOff = gAverage -  strikeGeometric;

            gPayOff = gPayOff >0 ? gPayOff  : 0.0f;

            *(outputDataGeometric + pathNumber) = gPayOff*df;
        }
    }
}




extern "C"
void AsianCallGPU_kernel_dev (thrust::device_vector<float>& input_normals_dev,
                              int normalsDimensions, // must be greater than or equal to stepsPerPath
                              int totalPaths, 
                              int stepsPerPath,
                              const thrust::host_vector<float>& logDrifts_host, 
                              const thrust::host_vector<float>& logSds_host, 
                              float logSpot,
                              float df,
                              float strikeArithmetic,
                              thrust::device_vector<float>& outputDataArithmetic,
                              float strikeGeometric,
                              thrust::device_vector<float>&outputDataGeometric)
{

    const int threadsperblock = 64;
    const int maxBlocks = 65535;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(totalPaths-1)/threadsperblock;

    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    // Fix the number of threads
    dimBlock.x = threadsperblock;


    thrust::device_vector<float> logDrifts_dev(logDrifts_host); 
    thrust::device_vector<float> logSds_dev(logSds_host); 

    COPYCONSTANTMEMORYFLOAT(dev_const_drifts,logDrifts_host);
    COPYCONSTANTMEMORYFLOAT(dev_const_logSds,logSds_host);


    CUT_CHECK_ERR("AsianGPU_kernel execution failed before entering kernel\n");

    thrust::device_vector<float> outputDataArithmeticDev(totalPaths);
    thrust::device_vector<float> outputDataGeometricDev(totalPaths);

    ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);

	Timer t1;

//    unsigned int hTimer;
//    double time;
//    cutilCheckError(cutCreateTimer(&hTimer));
 //   cutilCheckError(cutResetTimer(hTimer));
 //   cutilCheckError(cutStartTimer(hTimer));



    AsianCallGPU_transposed_kernel<<<dimGrid , dimBlock >>>(
        thrust::raw_pointer_cast(&input_normals_dev[0]),
        totalPaths, 
        stepsPerPath,
        logSpot,
        df,
        strikeArithmetic,
        thrust::raw_pointer_cast(&outputDataArithmeticDev[0]),
        strikeGeometric,
        thrust::raw_pointer_cast(&outputDataGeometricDev[0]));


   cutilSafeCall(cudaThreadSynchronize());
   // cutilCheckError(cutStopTimer(hTimer));
  //  time = cutGetTimerValue(hTimer);

    std::cout << " time taken for Asian kernel: " 
        << t1.timePassed() << std::endl;

    outputDataArithmetic = outputDataArithmeticDev;
    outputDataGeometric= outputDataGeometricDev;

    CUT_CHECK_ERR("AsianGPU_kernel execution failed\n");

}

