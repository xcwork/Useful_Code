

//
//
//                          early_exercise_value_generator_gpu.cu
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3

#include <early_exercise_value_generator_gpu.h>

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
                          )
{
    exerciseValueGenerator_gpu<earlyExerciseBermudanPayerSwaption>( exerciseValues_dev,  // output, one per exercise time per path
                               aux_float_data, // any float auxiliary data for the exercise value object 
                               aux_int_data, //  any int auxiliary data for the exercise value object 
                               exerciseIndices_dev, // the indices of the exercise times amongst evolution times
                               exerciseIndicators_dev, // boolean indicators of exercise amongst evolution times 
                               numberExerciseTimes,
                               pathsForOutput, // typically totalPaths
                               outputOffset,
                               batchPaths, 
                               numberStepsAndRates, // assumed to be equal 
                               rates1_dev, 
                               rates2_dev, 
                               rates3_dev, 
                               forwards_dev, 
                               discRatios_dev);

    
}
