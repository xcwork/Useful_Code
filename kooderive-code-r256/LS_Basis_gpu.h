//
//
//                  LS_basis_gpu.h
//
//
// (c) Mark Joshi 2011, 2013
// This code is released under the GNU public licence version 3



#ifndef LS_BASIS_GPU_H
#define LS_BASIS_GPU_H
#include <gpuCompatibilityCheck.h>
#include <gold/pragmas.h> 
#include <cudaWrappers/cudaTextureBinder.h>
#include <LS_basis_examples_gpu.h>




texture<int, 1, cudaReadModeElementType> tex_exerciseIndices;   // the indices of the exercise times amongst the evolution times
texture<int, 1, cudaReadModeElementType> tex_exerciseIndicators;  // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices

template<class T>
__global__ void basis_function_kernel(const int* __restrict__ integerData, // to access texture containing data specific to the basis variables
		                              const float* __restrict__ floatData, 
                                      int numberStepsAndRates,
                                      const float* __restrict__ forwards_dev,
                                      const float* __restrict__ discountRatios_dev,
                                      const float* __restrict__ rate1_dev,
                                      const float* __restrict__ rate2_dev,
                                      const float* __restrict__ rate3_dev,
                                      int numberPathsPerBatch,
                                      int totalNumberOfPaths, 
                                      int outputLayerSize,
                                      int batchLayerSize,
                                      int numberPathsPreviouslyDone, 
                                      int numberExerciseTimes,
                                      int exerciseIndices_offset, // the indices of the exercise times amongst the evolution times
                                      int exerciseIndicators_offset, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                                      float* basisFunctionVariables_dev // for output 
                                      )
{
    int bx = blockIdx.x;
    int tx =  threadIdx.x;
    
   
    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;
    
    int pathsPerThread = 1 + (( numberPathsPerBatch -1)/width);
    T obj(integerData, 
		  floatData, 
          numberStepsAndRates);

    
    for (int l=0; l < pathsPerThread; ++l)
    {
        int inputPathNumber = width*l + bwidth*bx+tx;

        int outputPathNumber = inputPathNumber+numberPathsPreviouslyDone;

        if (inputPathNumber < numberPathsPerBatch)
        {                
            for (int exTimeIndex =0; exTimeIndex < numberExerciseTimes; ++exTimeIndex)
            {
                int stepNumber = tex1Dfetch(tex_exerciseIndices,exerciseIndices_offset+exTimeIndex);

                int rateRateLoc = inputPathNumber+exTimeIndex*numberPathsPerBatch;

                float rate1f = rate1_dev[rateRateLoc];
                float rate2f = rate2_dev[rateRateLoc];
                float rate3f = rate3_dev[rateRateLoc];
                

                obj.writeBasisVariableValues(basisFunctionVariables_dev+outputPathNumber+outputLayerSize*exTimeIndex,
                                             outputLayerSize, // number variables*paths
		                                     stepNumber,  // the index amongst the rate times 
                                             exTimeIndex, // the index amongst the exercise times 
		                                     numberStepsAndRates, 
		                                     rate1f, 
		                                     rate2f, 
		                                     rate3f, 
                                             inputPathNumber,
		                                     forwards_dev, 
		                                     discountRatios_dev,
                                             totalNumberOfPaths,
                                             batchLayerSize);
                
            }               
        }
    }
}


template<class T>
void adjoinBasisVariables(int* integerData_dev, // to access texture containing data specific to the basis variables
		                  float* floatData_dev, // to access texture containing data specific to the basis variables
                          float* forwards_dev,
                          float* discountRatios_dev,
                          float* rate1_dev,
                          float* rate2_dev,
                          float* rate3_dev,
                          int numberPathsPerBatch,
                          int totalNumberOfPaths, 
                          int numberPathsPreviouslyDone,
                          int numberStepsAndRates, 
                          int numberExerciseTimes,
                          int* exerciseIndices_dev, // the indices of the exercise times amongst the evolution times
                          int* exerciseIndicators_dev, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                          float* basisFunctionVariables_dev // output location
                          )
{

    int outputLayerSize = T::maxVariablesPerStep()*totalNumberOfPaths;

//    int offsetIntegerData=0;
//    int offsetFloatData=0;
    int exerciseIndices_offset=0;
    int exerciseIndicators_offset=0;
    
//    cudaTextureFloatBinder basisFloatDataBinderWrapper(tex_basis_var_aux_float_data,floatData_dev);
//    cudaTextureIntBinder basisIntDataBinderWrapper(tex_basis_var_aux_int_data,integerData_dev);

    cudaTextureIntBinder exIndicesBinderWrapper(tex_exerciseIndices,exerciseIndices_dev);   // the indices of the exercise times amongst the evolution times
    cudaTextureIntBinder exIndicatorsBinderWrapper(tex_exerciseIndicators,exerciseIndicators_dev);

      const int threadsperblock = 64;
    const int maxBlocks = 65535;
    
    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(numberPathsPerBatch-1)/threadsperblock;
    
    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;
  
    // Fix the number of threads
    dimBlock.x = threadsperblock;
       ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);

    basis_function_kernel<T><<<dimGrid , dimBlock >>>(integerData_dev, 
		                  floatData_dev, 
                          numberStepsAndRates,
                          forwards_dev,
                          discountRatios_dev,
                          rate1_dev,
                          rate2_dev,
                          rate3_dev,
                          numberPathsPerBatch,
                          totalNumberOfPaths, 
                          outputLayerSize,
                          numberPathsPerBatch*numberStepsAndRates,
                          numberPathsPreviouslyDone, 
                          numberExerciseTimes,
                          exerciseIndices_offset, // the indices of the exercise times amongst the evolution times
                          exerciseIndicators_offset, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                          basisFunctionVariables_dev // for output 
                                      );
}

// non template function using example basis set
// note it is "adjoin" not "adjoint"! 
extern "C"
void adjoinBasisVariablesCaller_gpu(bool useLogs,
									int* integerData_dev, // to access texture containing data specific to the basis variables
		                  float* floatData_dev, // to access texture containing data specific to the basis variables
                          float* forwards_dev,
                          float* discountRatios_dev,
                          float* rate1_dev,
                          float* rate2_dev,
                          float* rate3_dev,
                          int numberPathsPerBatch,
                          int totalNumberOfPaths, 
                          int numberPathsPreviouslyDone,
                          int numberStepsAndRates, 
                          int numberExerciseTimes,
                          int* exerciseIndices_dev, // the indices of the exercise times amongst the evolution times
                          int* exerciseIndicators_dev, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                          float* basisFunctionVariables_dev // output location
                          );

#endif
