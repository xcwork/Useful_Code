//
//
//                  LS_gpu.h
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3



#ifndef LS_GPU_H
#define LS_GPU_H
#include <gpuCompatibilityCheck.h>
#include <gold/pragmas.h> 
#include <cudaWrappers/cudaTextureBinder.h>



// non template function using example basis set

void expandBasisFunctions_quadratic_gpu(
                                        int totalNumberOfPaths, 
                                        float* basisVariables_global, // input location for basis variables for all steps,
                                        int exerciseNumber,
                                        int maxBasisVariableSize, 
                                        int variableSizeThisStep,
								        bool useCrossTerms,
                                        float* basisFunctions_global
                                        );
// compute continuation values and subtract exercise values

double continuation_net_value_evaluation_gpu(int paths,
                           int exerciseNumber,
                           float* basisFunctions_global, // just the function values for this step
                           int numberBasisFunctions,
                           float* basis_weights_global, // the weights for thsi regression
                            float* exerciseValues_global,      // the exerciseValues for all steps deflated to  exercise times
                           float AndersenShift,
						   float* output_global);


// this does the exercise decisions for one step of the first pass of an LS regression for early exercise
/*
double oneStepUpdateLS_gpu(int paths,
                   int exerciseNumber,
                   float* basisFunctions_global, // just the function values for this step
                   int numberBasisFunctions,
                   float* basis_weights_global, // the weights for all steps
                   float* continuationValues_global,  //the continuation values at the moment, this will be overwritten, deflated to current ex time
                   float* deflatedCashFlows_global,   // the cash-flow values all steps deflated to ex times
                   float* exerciseValues_global,      // the exerciseValues for all steps deflated to  exercise times
                   float AndersenShift);
				   */

// this does the exercise decisions for one step of the first pass of an LS multiple regression for early exercise
double oneStepUpdateLS_multi_gpu(int paths,
                           int exerciseNumber,
                           float* basisFunctions_global, // just the function values for this step
                           int numberBasisFunctions,
						   int basisFunctionsRowSize,
						   int maxRegressionDepth,
                           float* basis_weights_global, // the weights for all steps and regression depths, 
						   float* lowerCuts_global,
						   float* upperCuts_global,
                           float* continuationValues_global,  //the continuation values at the moment, this will be overwritten, deflated to current ex time
                           float* deflatedCashFlows_global,   // the cash-flow values all steps deflated to ex times
                           float* exerciseValues_global,      // the exerciseValues for all steps deflated to  exercise times
                           float AndersenShift,
						   float* estContinuationValues_global,
						   int* exercises_global );

// returns time taken
// overwrites deflatedNextStepValues_global
// numeraireValues_global is matrix of form exercise dates \times paths 

 double updateDeflationOfContinuationValues_gpu( float* deflatedNextStepValues_global,  
                                             float* numeraireValues_global,
                                             int paths,
                                             int newStepNumber,
                                             int oldStepNumber);




texture<float, 1, cudaReadModeElementType> tex_basis_var_aux_float_data;
texture<int, 1, cudaReadModeElementType> tex_basis_var_aux_int_data;


/* Example of behaviours required of T*/

// note only does 1 x y z x^2 y^2 z^2 , i.e. no cross terms 
class quadraticPolynomialDevice
{
private:

    int numberVariables_;

public:

    __device__  quadraticPolynomialDevice(	int offsetIntegerData, // to access texture containing data specific to the basis variables
        int offsetFloatData, 
        int numberVariables) :numberVariables_(numberVariables)
    {
    }

    __device__ void writeBasisFunctionsValues(
        float* input_global_location,        
        float* output_global_location, // includes pathNumber 
        int totalPaths
        )
    {
        output_global_location[0] = 1.0f;
		int j=1;

        for (int i=0; i < numberVariables_; ++i, ++j)
        {
            float x= input_global_location[i*totalPaths];
            output_global_location[j*totalPaths] = x;
			++j;
            output_global_location[j*totalPaths]=x*x;
        }



    }

    __host__ static int functionValues(int numberVariables)
    {   
        return numberVariables*2+1;
    }


};
/* Example of behaviours required of T*/

// note does 1 x y z x^2 y^2 z^2 xy yz zx ie has cross terms between each variable and its successor 
// this gives all cross terms for 3 variables but would miss some with 4 or more
class quadraticPolynomialCrossDevice
{
private:

    int numberVariables_;

public:

    __device__  quadraticPolynomialCrossDevice(	int offsetIntegerData, // to access texture containing data specific to the basis variables
        int offsetFloatData, 
        int numberVariables) :numberVariables_(numberVariables)
    {
    }

    __device__ void writeBasisFunctionsValues(
        float* input_global_location,        
        float* output_global_location, // includes pathNumber 
        int totalPaths
        )
    {
        output_global_location[0] = 1.0f;

		int j=1;

        for (int i=0; i < numberVariables_; ++i,++j)
        {
            float x= input_global_location[i*totalPaths];
			output_global_location[j*totalPaths] = x;
		}

		if (numberVariables_ ==1)
		{
			float x=input_global_location[0];
			output_global_location[j*totalPaths]  = x*x;
			return;
		}

		if (numberVariables_==2)
		{
			float x=input_global_location[0];
			float y=input_global_location[totalPaths];
			output_global_location[j*totalPaths]  = x*x;
			++j;
			output_global_location[j*totalPaths]  = x*y;
			++j;
			output_global_location[j*totalPaths]  = y*y;
			return;
		}

		for (int i=0; i < numberVariables_; ++i,++j)
		{
			float x =input_global_location[i*totalPaths];
		    int k = (i+1) % numberVariables_;
			float y = input_global_location[k*totalPaths];
			output_global_location[j*totalPaths] = x*y;

		}
		

		for (int i=0; i < numberVariables_; ++i,++j)
		{
			float x =input_global_location[i*totalPaths];
		    
			output_global_location[j*totalPaths] = x*x;

		}

			

    }

    __host__ static int functionValues(int numberVariables)
    {   
		int total;

        if (numberVariables ==0)
			total = 1;
		else
			if (numberVariables==1)
				total = 3;
			else
				if (numberVariables ==2)
					total = 6;
				else
					total=1+3*numberVariables;

		return total;
	}
    


};




texture<float, 1, cudaReadModeElementType> tex_basis_functions_aux_float_data;
texture<int, 1, cudaReadModeElementType> tex_basis_functions_aux_int_data;


template<class T>
__global__ void basis_function_expansion_kernel(int offsetIntegerData, // to access texture containing data specific to the basis variables
                                                int offsetFloatData, 
                                                int totalNumberOfPaths, 
                                                int numberVariables,
                                                float* input_global_location,
                                                float* output_global_location // for output 
                                                )
{
    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;

    int pathsPerThread = 1 + (( totalNumberOfPaths -1)/width);
    T obj(offsetIntegerData, // to access texture containing data specific to the basis variables
        offsetFloatData, 
        numberVariables);


    for (int l=0; l < pathsPerThread; ++l)
    {
        int inputPathNumber = width*l + bwidth*bx+tx;

        if (inputPathNumber < totalNumberOfPaths)                    
            obj.writeBasisFunctionsValues(input_global_location+inputPathNumber,        
            output_global_location+inputPathNumber, 
            totalNumberOfPaths);

    }
}


template<class T>
void expandBasisFunctions(int* integerData_dev, // to access texture containing data specific to the basis variables
                          float* floatData_dev,
                          int totalNumberOfPaths, 
                          float* basisVariables_global, // input location for basis variables for all steps,
                          int exerciseNumber,
                          int maxBasisVariableSize, 
                          int variableSizeThisStep,
                          float* basisFunctions_global // output location
                          )
{


    int offsetIntegerData=0;
    int offsetFloatData=0;

    cudaTextureFloatBinder basisFloatDataBinderWrapper(tex_basis_var_aux_float_data,floatData_dev);
    cudaTextureIntBinder basisIntDataBinderWrapper(tex_basis_var_aux_int_data,integerData_dev);

    const int threadsperblock = 64;
    const int maxBlocks = 65535;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(totalNumberOfPaths-1)/threadsperblock;

    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    // Fix the number of threads
    dimBlock.x = threadsperblock;

    float* input_global_location = basisVariables_global + totalNumberOfPaths*maxBasisVariableSize*exerciseNumber;

    ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);
    
    basis_function_expansion_kernel<T><<<dimGrid , dimBlock >>>(offsetIntegerData, // to access texture containing data specific to the basis variables
        offsetFloatData, 
        totalNumberOfPaths,
        variableSizeThisStep,
        input_global_location,
        basisFunctions_global);


}



#endif
