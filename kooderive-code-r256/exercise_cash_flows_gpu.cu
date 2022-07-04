//
//
//                                   exercise_cash_flows_gpu.cu
//
//
//

// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3

// #include "cashFlowDiscounting_gpu.h"


#include <cudaWrappers/cudaTextureBinder.h>

#include <gpuCompatibilityCheck.h>
/*
if exercise time is j < numberSteps
set cashFlows[j] to exerciseValues[j]

*/
__global__ void exercise_cash_flows_kernel(float* genFlows,  // gen flows will be modified
										   int* firstIndex,
										   int* secondIndex,
										   float* theta, 
										   float* exerciseValues,
										   int* firstIndexExercise,
										   int* secondIndexExercise,
										   float* thetaExercise,
										   int* exerciseTimes, 
										   int paths, 
										   int numberSteps )
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
        int dataOffset = pathNumber;

        
        if (pathNumber < paths)
        {
			int exTime =  exerciseTimes[pathNumber];
			for (int i=0; i < numberSteps;++i)
			{
				int off = dataOffset+i*paths;
				if (i==exTime)
				{
					
					genFlows[off]=exerciseValues[off];
					firstIndex[off] = firstIndexExercise[off];
					secondIndex[off] = secondIndexExercise[off];
					theta[off] = thetaExercise[off];
				}
				else
					if (i > exTime)
						genFlows[off] =0.0f;
			}
                   
                  
        }
    }
}


                                                                  

extern"C"
void exercise_cash_flows_gpu(float* genFlows,  // gen flows will be modified
										   int* firstIndex, // will be changed
										   int* secondIndex, // will be changed
										   float* theta,  // will be changed
										   float* exerciseValues,
										   int* firstIndexExercise,
										   int* secondIndexExercise,
										   float* thetaExercise,
										   int* exerciseTimes, 
										   int paths, 
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
   
            
    ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);

    exercise_cash_flows_kernel<<<dimGrid , dimBlock >>>(genFlows,  // gen flows will be modified
											 firstIndex,
											 secondIndex,
										     theta, 
										     exerciseValues,
										     firstIndexExercise,
										     secondIndexExercise,
										     thetaExercise,
										     exerciseTimes, 
										     paths, 
										     numberSteps);
}
