


//
//
//                          exercise_values_examples.h
//
//
// (c) Mark Joshi 2011,2013
// This code is released under the GNU public licence version 3

/*
a couple of basic examples of early exercise values 

*/

#ifndef EXERCISE_VALUES_H
#define EXERCISE_VALUES_H

texture<float, 1, cudaReadModeElementType> tex_exvalue_aux_float_data;
texture<int, 1, cudaReadModeElementType> tex_exvalue_aux_int_data;


class earlyExerciseNull
{
public:
    __device__  earlyExerciseNull(int floatDataOffset,
        int intDataOffset, 
        int numberOfRatesAndSteps)
    {}

    __device__ float exerciseValue(int stepEvolution,
        int exerciseTimeIndex,
        float rate1,
        float rate2,
        float rate3,
        int pathNumber, // needed to find right bit of forwards and discratios
        int numberOfPaths, // needed to find right bit of forwards and discratios
        const float* __restrict__  forwards_dev,
        const float* __restrict__  discRatios_dev)
    {   
        return 0.0f;
    }


};


class earlyExerciseBermudanPayerSwaption
{
private:
    float strike_;
public:
    __device__   earlyExerciseBermudanPayerSwaption(int floatDataOffset,
        int intDataOffset, 
        int numberOfRatesAndSteps)
    {
        strike_ = tex1Dfetch(tex_exvalue_aux_float_data,floatDataOffset);
    }

    __device__  float exerciseValue(int stepEvolution,
        int exerciseTimeIndex,
        float rate1, // interpreted as coterminal swap rate
        float rate2, // interpreted as coterminal annuity
        float rate3, // ignored 
        int pathNumber, // needed to find right bit of forwards and discratios
        int numberOfPaths, // needed to find right bit of forwards and discratios
        const float* __restrict__ forwards_dev,
        const float* __restrict__ discRatios_dev)
    {   
        float swapVal = (rate1-strike_)*rate2;

        return swapVal > 0.0f ? swapVal : 0.0f;
    }


};
#endif
