//
//
//                  LS_basis_examples_gpu.h
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3



#ifndef LS_BASIS_EXAMPLES_GPU_H
#define LS_BASIS_EXAMPLES_GPU_H

#include <gold/pragmas.h> 



//texture<float, 1, cudaReadModeElementType> tex_basis_var_aux_float_data;
//texture<int, 1, cudaReadModeElementType> tex_basis_var_aux_int_data;

/* 
Example of behaviours required of basis function classes
note that the caller specifies rate1, rate2, rate3 so the choice of these rates is done elsewhere
they can also be post processed into polynomials elsewhere 

*/

class basisVariableExample
{
private:

    int numberStepsAndRates_;

public:

__device__  basisVariableExample(	const int* __restrict__ integerData, // to access texture containing data specific to the basis variables
		                const float* __restrict__ floatData, 
                        int numberStepsAndRates) :numberStepsAndRates_(numberStepsAndRates)
{
}

__device__ void writeBasisVariableValues(float* output_global_location, // includes pathNumber and stepNumber offset 
        int layerSize, // 
		int numberOfRatesReset,  // the index amongst the rate times 
        int exerciseIndex, // the index amongst the exercise times 
		int numberRates, 
		float rate1, 
		float rate2, 
		float rate3, 
        int inputPathNumber,
		const float* __restrict__ allForwardRates, 
		const float* __restrict__ allDiscountRatios,
        int totalPaths,
        int batchLayerSize
		)
{
    float variable1 = rate1;
    float variable2 = rate2;
    float variable3 = rate3;
    *output_global_location=variable1;
    output_global_location[totalPaths]=variable2;
    output_global_location[2*totalPaths]=variable3;

}

__host__ static int maxVariablesPerStep()
{   
    return 3;
}
__host__ static int actualNumberOfVariables(int step,
											   int stepIndexAmongstRates,
											   int numberRates)
{
	if (stepIndexAmongstRates == numberRates-1)
		return 1;
	if (stepIndexAmongstRates == numberRates-2)
		return 2;

	return 3;

}


};
class basisVariableLog_gpu
{
private:

    int numberStepsAndRates_;
	float shift1;
	float shift2;
	float shift3;

public:

__device__  basisVariableLog_gpu(	const int* __restrict__ integerData, // to access texture containing data specific to the basis variables
		                const float* __restrict__ floatData, 
                        int numberStepsAndRates) :numberStepsAndRates_(numberStepsAndRates)
{
	shift1 = floatData[0];
	shift2 = floatData[1];
	shift3 = floatData[2];
}

__device__ void writeBasisVariableValues(float* output_global_location, // includes pathNumber and stepNumber offset 
        int layerSize, // 
		int numberOfRatesReset,  // the index amongst the rate times 
        int exerciseIndex, // the index amongst the exercise times 
		int numberRates, 
		float rate1, 
		float rate2, 
		float rate3, 
        int inputPathNumber,
		const float* __restrict__ allForwardRates, 
		const float* __restrict__ allDiscountRatios,
        int totalPaths,
        int batchLayerSize
		)
{
    float variable1 = log(rate1+shift1);
    float variable2 = log(rate2+shift2);
    float variable3 = log(rate3+shift3);
    *output_global_location=variable1;
    output_global_location[totalPaths]=variable2;
    output_global_location[2*totalPaths]=variable3;

}

__host__ static int maxVariablesPerStep()
{   
    return 3;
}
__host__ static int actualNumberOfVariables(int step,
											   int stepIndexAmongstRates,
											   int numberRates)
{
	if (stepIndexAmongstRates == numberRates-1)
		return 1;
	if (stepIndexAmongstRates == numberRates-2)
		return 2;

	return 3;

}


};
#endif
