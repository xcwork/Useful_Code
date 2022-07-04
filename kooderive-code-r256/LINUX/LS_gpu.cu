//
//
//                  LS_gpu.cu
//
//
// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3


#include <LS_gpu.h>
#include <thrust/device_vector.h>
#include <cutil.h>

#include <cutil_inline.h>
#include <cuda_runtime.h>

void expandBasisFunctions_quadratic_gpu(
                                        int totalNumberOfPaths, 
                                        float* basisVariables_global, // input location for basis variables for all steps,
                                        int exerciseNumber,
                                        int maxBasisVariableSize, 
                                        int variableSizeThisStep,
										bool useCrossTerms,
                                        float* basisFunctions_global
                                        )
{

    thrust::device_vector<int> integer_data_vec(1);
    integer_data_vec[0] = variableSizeThisStep;
    thrust::device_vector<float> float_data_vec(1);
    float_data_vec[0] = 0.0f;


	if (useCrossTerms)
		expandBasisFunctions<quadraticPolynomialCrossDevice>( thrust::raw_pointer_cast(&integer_data_vec[0]), // to access texture containing data specific to the basis variables
			thrust::raw_pointer_cast(&float_data_vec[0]),                  
			totalNumberOfPaths, 
			basisVariables_global, // input location for basis variables for all steps,
			exerciseNumber,
			maxBasisVariableSize, 
			variableSizeThisStep,
			basisFunctions_global // output location
			);
	else
		expandBasisFunctions<quadraticPolynomialDevice>( thrust::raw_pointer_cast(&integer_data_vec[0]), // to access texture containing data specific to the basis variables
			thrust::raw_pointer_cast(&float_data_vec[0]),                  
		    totalNumberOfPaths, 
            basisVariables_global, // input location for basis variables for all steps,
            exerciseNumber,
            maxBasisVariableSize, 
            variableSizeThisStep,
            basisFunctions_global // output location
            );
}


__global__ void continuation_net_value_evaluation_kernel(int paths, 
                                         int numberBasisFunctions,
                                         float* basis_weights_global, 
                                         float* basis_functions_values_global,
                       //                  float* continuationValues_global,
                                         float* exerciseValues_global,
                                         float AndersenShift,
										 float* outputs_global)

{
    extern __shared__ float basis_weights_shared[];

    // put basis weights into shared memory for faster access

    float* weights_s = basis_weights_shared;

    int tx = threadIdx.x;

    while (tx < numberBasisFunctions)
    {
        weights_s[tx] = basis_weights_global[tx];

        tx += blockDim.x;
    }
    __syncthreads();


    int path0 = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int path = path0; path < paths; path += stride)
    {
        float estContValue = AndersenShift;
        for (int i=0; i < numberBasisFunctions;++i)
            estContValue+= weights_s[i]*basis_functions_values_global[path+i*paths];

        float exValue = exerciseValues_global[path];

		outputs_global[path] = estContValue - exValue;

   
    }
}

double continuation_net_value_evaluation_gpu(int paths,
                           int exerciseNumber,
                           float* basisFunctions_global, // just the function values for this step
                           int numberBasisFunctions,
                           float* basis_weights_global, // the weights for this regression
                    //       float* continuationValues_global,  //the continuation values at the moment, this will be overwritten, deflated to current ex time
                            float* exerciseValues_global,      // the exerciseValues for this step deflated to  exercise times
                           float AndersenShift,
						   float* output_global)
{

    int size = numberBasisFunctions*sizeof(float);

    const int threadsperblock = 64;
    const int maxBlocks = 65535;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;

    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    // Fix the number of threads
    dimBlock.x =threadsperblock;

    cutilSafeCall(cudaThreadSynchronize());

	Timer h1;


    continuation_net_value_evaluation_kernel<<<dimGrid, dimBlock,size>>>(paths, 
        numberBasisFunctions,
        basis_weights_global,
        basisFunctions_global,
  //      continuationValues_global,
        exerciseValues_global, //+paths*exerciseNumber,
        AndersenShift,
		output_global);

    cutilSafeCall(cudaThreadSynchronize());

    double time = h1.timePassed();

    return time;

}

/*
__global__ void exercise_updating_LS_kernel(int paths, 
                                         int numberBasisFunctions,
                                         float* basis_weights_global, 
                                         float* basis_functions_values_global,
                                         float* continuationValues_global,
                                         float* deflatedCashFlows_global,
                                         float* exerciseValues_global,
                                         float AndersenShift)

{
    extern __shared__ float basis_weights_shared[];

    // put basis weights into shared memory for faster access

    float* weights_s = basis_weights_shared;

    int tx = threadIdx.x;

    while (tx < numberBasisFunctions)
    {
        weights_s[tx] = basis_weights_global[tx];

        tx += blockDim.x;
    }
    __syncthreads();


    int path0 = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int path = path0; path < paths; path += stride)
    {
        float estContValue = AndersenShift;
        for (int i=0; i < numberBasisFunctions;++i)
            estContValue+= weights_s[i]*basis_functions_values_global[path+i*paths];

        float exValue = exerciseValues_global[path];

        if (estContValue < exValue)
            continuationValues_global[path] = exValue;
   //     else // these have already been added elsewhere
     //       continuationValues_global[path] += deflatedCashFlows_global[path];
    }
}
*/

/*
double oneStepUpdateLS_gpu(int paths,
                           int exerciseNumber,
                           float* basisFunctions_global, // just the function values for this step
                           int numberBasisFunctions,
                           float* basis_weights_global, // the weights for all steps
                           float* continuationValues_global,  //the continuation values at the moment, this will be overwritten, deflated to current ex time
                           float* deflatedCashFlows_global,   // the cash-flow values all steps deflated to ex times
                           float* exerciseValues_global,      // the exerciseValues for all steps deflated to  exercise times
                           float AndersenShift)
{

    int size = numberBasisFunctions*sizeof(float);

    const int threadsperblock = 64;
    const int maxBlocks = 65535;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;

    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    // Fix the number of threads
    dimBlock.x =threadsperblock;

    cutilSafeCall(cudaThreadSynchronize());

	Timer h1;


    exercise_updating_LS_kernel<<<dimGrid, dimBlock,size>>>(paths, 
        numberBasisFunctions,
        basis_weights_global+numberBasisFunctions*exerciseNumber,
        basisFunctions_global,
        continuationValues_global,
        deflatedCashFlows_global+paths*exerciseNumber,
        exerciseValues_global+paths*exerciseNumber,
        AndersenShift);

    cutilSafeCall(cudaThreadSynchronize());

    double time = h1.timePassed();

    return time;

}

*/

__global__ void numeraire_deflating_kernel(float* deflatedNextStepValues_global, 
                                           float* numeraireValues1_global,
                                           float* numeraireValues2_global,
                                           int paths)

{


    int path0 = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int path = path0; path < paths; path += stride)
    {
        float df = numeraireValues1_global[path]/numeraireValues2_global[path];
        deflatedNextStepValues_global[path] *= df;

    }
}

 double updateDeflationOfContinuationValues_gpu( float* deflatedNextStepValues_global, 
                                             float* numeraireValues_global,
                                             int paths,
                                             int newStepNumber,
                                             int oldStepNumber)
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
    dimBlock.x =threadsperblock;

    cutilSafeCall(cudaThreadSynchronize());

  Timer h1;

    numeraire_deflating_kernel<<<dimGrid, dimBlock>>>(deflatedNextStepValues_global, 
        numeraireValues_global+newStepNumber*paths,
        numeraireValues_global+oldStepNumber*paths,
         paths);

    cutilSafeCall(cudaThreadSynchronize());

  
    double time = h1.timePassed();

    return time;
}
////////////////


 

__global__ void exercise_updating_LS_multi_kernel(int paths, 
                                         int numberBasisFunctions,
										 int basisFunctionsRowSize,
										 int maxRegressionDepth,
                                         float* basis_weights_global, 
										 float* lower_cuts_global,
										 float* upper_cuts_global,
                                         float* basis_functions_values_global,
                                         float* continuationValues_global,
                                         float* deflatedCashFlows_global,
                                         float* exerciseValues_global,
                                         float AndersenShift,
										 float* estContinuationValues,
										 int* exercises)

{
    extern __shared__ float basis_weights_shared[];

    // put basis weights and cut points into shared memory for faster access

	float* lower_cut_s = basis_weights_shared;
	float* upper_cut_s = basis_weights_shared+maxRegressionDepth;
  //  float* weights_s = basis_weights_shared+2*maxRegressionDepth;

    int tx = threadIdx.x;

 /*   while (tx < basisFunctionsRowSize*maxRegressionDepth)
    {
        weights_s[tx] = basis_weights_global[tx];

        tx += blockDim.x;
    }
	*/
	tx = threadIdx.x;

    while (tx < maxRegressionDepth)
    {
        lower_cut_s[tx] = lower_cuts_global[tx];
        upper_cut_s[tx] = upper_cuts_global[tx];
        tx += blockDim.x;
    }
    __syncthreads();


    int path0 = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int path = path0; path < paths; path += stride)
    {
		bool stillGoing=true;
		bool exercise=false;
		int depth =0;
		float estContValue =0.0f;
		float exValue = exerciseValues_global[path];
		float* thisWeights_s = basis_weights_global;
			//weights_s;
		
		while(stillGoing)
		{
	        estContValue = AndersenShift;
		    for (int i=0; i < numberBasisFunctions;++i)
			    estContValue+= thisWeights_s[i]*basis_functions_values_global[path+i*paths];

			float netValue = estContValue-exValue;

			if (depth==maxRegressionDepth-1 ||  netValue < lower_cut_s[depth] || netValue > upper_cut_s[depth] )
			{
				stillGoing=false;
			
				exercise = netValue < 0.0f;
			}
			else
			{
				++depth;	
				thisWeights_s += basisFunctionsRowSize;
			}
		}

        if (exercise)
		{
            continuationValues_global[path] = exValue;
		
		}
		exercises[path] = exercise ? 1 : -1;
		estContinuationValues[path]=estContValue;

   
    }
}



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
						   int* exercises_global )
{

    int size = (numberBasisFunctions+2)*maxRegressionDepth*sizeof(float);

    const int threadsperblock = 64;
    const int maxBlocks = 65535;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    dimGrid.x = 1+(paths-1)/threadsperblock;

    if (dimGrid.x > maxBlocks) 
        dimGrid.x=maxBlocks;

    // Fix the number of threads
    dimBlock.x =threadsperblock;

    cutilSafeCall(cudaThreadSynchronize());


	float* basis_weights_loc = basis_weights_global+basisFunctionsRowSize*exerciseNumber*maxRegressionDepth;

	
	Timer h1;


    exercise_updating_LS_multi_kernel<<<dimGrid, dimBlock,size>>>(paths, 
        numberBasisFunctions,
		basisFunctionsRowSize,
		maxRegressionDepth,
        basis_weights_loc,
		lowerCuts_global,
		upperCuts_global,
        basisFunctions_global,
        continuationValues_global,
        deflatedCashFlows_global+paths*exerciseNumber,
        exerciseValues_global+paths*exerciseNumber,
        AndersenShift,
		estContinuationValues_global,
		exercises_global
		);

    cutilSafeCall(cudaThreadSynchronize());

    double time = h1.timePassed();

    return time;

}

