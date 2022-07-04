

//
//
//                          early_exercise_value_generator_gpu.h
//
//
// (c) Mark Joshi 2011,2013
// This code is released under the GNU public licence version 3

/*
Routine to generate the exercise values given paths. 
Inputs are batched. 
Outputs are collated. 

Note that the timing of payments is not addressed here.
It is intended that a separate class will take in the exercise values 
and payment schedule and deflate them to exercise decision days. 

*/

#ifndef EARLY_EXERCISE_VALUE_GENERATOR_GPU_H
#define EARLY_EXERCISE_VALUE_GENERATOR_GPU_H


#include <gold/pragmas.h> 
#include <cudaWrappers/cudaTextureBinder.h>
#include <exercise_values_examples.h>


texture<int, 1, cudaReadModeElementType> tex_exerciseIndices;   // the indices of the exercise times amongst the evolution times
texture<int, 1, cudaReadModeElementType> tex_exerciseIndicators;  // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices




template<class T>
__global__ void exercise_value_kernel(int offsetIntegerData, // to access texture containing data specific to the exercise value object
                                      int offsetFloatData, // to access texture containing data specific to the  exercise value object
                                      int numberStepsAndRates,
                                      float* forwards_dev,
                                      float* discountRatios_dev,
                                      float* rate1_dev,
                                      float* rate2_dev,
                                      float* rate3_dev,
                                      int numberPathsPerBatch,
                                      int totalNumberOfPaths, 
                                      int numberPathsPreviouslyDone, 
                                      int numberExerciseTimes,
                                      int exerciseIndices_offset, // the indices of the exercise times amongst the evolution times
                                      int exerciseIndicators_offset, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
                                      float* exercise_values_dev // for output 
                                      )
{
    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( numberPathsPerBatch -1)/width);
    T obj(offsetIntegerData, // to access texture containing data specific to the exercise values
        offsetFloatData, 
        numberStepsAndRates);


    for (int l=0; l < pathsPerThread; ++l)
    {
        int inputPathNumber = width*l + bwidth*bx+tx;

 
        if (inputPathNumber < numberPathsPerBatch)
        {                
            for (int exTimeIndex =0; exTimeIndex < numberExerciseTimes; ++exTimeIndex)
            {
                int stepNumber =  tex1Dfetch(tex_exerciseIndices,exerciseIndices_offset+exTimeIndex);

                int rateRateLoc = inputPathNumber+exTimeIndex*numberPathsPerBatch;
                int outputLoc = inputPathNumber+numberPathsPreviouslyDone+exTimeIndex*totalNumberOfPaths;

                float rate1f = rate1_dev[rateRateLoc];
                float rate2f = rate2_dev[rateRateLoc];
                float rate3f = rate3_dev[rateRateLoc];

                float exValue = obj.exerciseValue(stepNumber,
                    exTimeIndex,
                    rate1f, 
                    rate2f, 
                    rate3f, 
                    inputPathNumber, // needed to find right bit of forwards and discratios
                    numberPathsPerBatch, // needed to find right bit of forwards and discratios
                    forwards_dev,
                    discountRatios_dev);

                exercise_values_dev[outputLoc] = exValue;





            }               
        }
    }
}

//routine that calls the exercise value kernel given that 
//the data is already on the gpu


template<class T> 
void 
exerciseValueGenerator_gpu(float* exerciseValues_dev,  // output, one per exercise time per path
                           float* aux_float_data, // any float auxiliary data for the exercise value object 
                           int* aux_int_data, //  any int auxiliary data for the exercise value object 
                           int* exerciseIndices_dev, // the indices of the exercise times amongst evolution times
                           int* exerciseIndicators_dev, // boolean indicators of exercise amongst evolution times 
                           int numberExerciseTimes,
                           int pathsForOutput, // typically totalPaths
                           int outputOffset,
                           int batchPaths, 
                           int numberStepsAndRates, // assumed to be equal 
                           float* rates1_dev, 
                           float* rates2_dev, 
                           float* rates3_dev, 
                           float* forwards_dev, 
                           float* discRatios_dev)
{


    int offsetIntegerData=0;
    int offsetFloatData=0;
    int exerciseIndices_offset=0;
    int exerciseIndicators_offset=0;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    // set texture parameters
    tex_exvalue_aux_float_data.addressMode[0] = cudaAddressModeWrap;
    tex_exvalue_aux_float_data.addressMode[1] = cudaAddressModeWrap;
    tex_exvalue_aux_float_data.filterMode = cudaFilterModeLinear;
    tex_exvalue_aux_float_data.normalized = false;    // access with normalized texture coordinates
    cudaBindTexture( NULL, tex_exvalue_aux_float_data, aux_float_data, channelDesc);

    cudaBindTexture( NULL, tex_exvalue_aux_int_data, aux_int_data);

    cudaTextureIntBinder exIndicesBinderWrapper(tex_exerciseIndices,exerciseIndices_dev);   // the indices of the exercise times amongst the evolution times
    cudaTextureIntBinder exIndicatorsBinderWrapper(tex_exerciseIndicators,exerciseIndicators_dev);

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

    exercise_value_kernel<T><<<dimGrid , dimBlock >>>(offsetIntegerData, // to access texture containing data specific to the exercise value object
        offsetFloatData, // to access texture containing data specific to the  exercise value object
        numberStepsAndRates,
        forwards_dev,
        discRatios_dev,
        rates1_dev,
        rates2_dev,
        rates3_dev,
        batchPaths,
        pathsForOutput, 
        outputOffset, 
        numberExerciseTimes,
        exerciseIndices_offset, // the indices of the exercise times amongst the evolution times
        exerciseIndicators_offset, // at each evol time, 1 if exercisable, 0 if not. Same information contents as exerciseIndices
        exerciseValues_dev // for output 
        );


    cudaUnbindTexture(tex_exvalue_aux_int_data);
    cudaUnbindTexture(tex_exvalue_aux_float_data);



}


// non template function using example exercise values  of Bermudan swaption class
// note it is "adjoin" not "adjoint"! 
extern "C"
void adjoinExerciseValues_Bermudan_swaption_gpu(float* exerciseValues_dev,  // output, one per exercise time per path
                                                float* aux_float_data, // any float auxiliary data for the exercise value object 
                                                int* aux_int_data, //  any int auxiliary data for the exercise value object 
                                                int* exerciseIndices_dev, // the indices of the exercise times amongst evolution times
                                                int* exerciseIndicators_dev, // boolean indicators of exercise amongst evolution times 
                                                int numberExerciseTimes,
                                                int pathsForOutput, // typically totalPaths
                                                int outputOffset,
                                                int batchPaths, 
                                                int numberStepsAndRates, // assumed to be equal 
                                                float* rates1_dev, 
                                                float* rates2_dev, 
                                                float* rates3_dev, 
                                                float* forwards_dev, 
                                                float* discRatios_dev
                                                );


#endif
