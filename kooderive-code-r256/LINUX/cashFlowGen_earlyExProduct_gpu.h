//
//                                  cashFlowGen_earlyEx_product_gpu.h
//
//


// (c) Mark Joshi 2013
// This code is released under the GNU public licence version 3

#ifndef CASH_FLOW_GEN_EARLY_EX_PROD_GPU_H
#define CASH_FLOW_GEN_EARLY_EX_PROD_GPU_H

#include <cashFlowGeneration_product_gpu.h>
#include <cashFlowGeneration_gpu.h>
#include <exercise_values_examples.h>
#include <vector>
#include <gold/MatrixFacade.h> 
#include <algorithm>

#include <cudaMacros.h>

/*
R is original product to be broken, eg could be zero or a swap
S is exercise value on breaking, eg a rebate of zero, or the pay-off of a Bermudan swaption
T is exercise strategy eg the LSA strategy
*/

template<class R, class S, class T>
__global__ void cashFlowGeneratorEE_kernel(float* genFlows1_global,
										   float*  genFlows2_global, 
										   int productDataTextureOffset,
										   int exerciseValueFloat_tex_offset,
										   int exerciseValueInt_tex_offset,
										   float*  exerciseStrategyDataFloat_global,
										   int exerciseStrategyDataFloat_size,
										   int*  exerciseStrategyDataInt_global,
										   int  exerciseStrategyDataInt_size,
										   int paths, 
										   int numberSteps,
										   int numberExerciseDates, 
										   const bool* __restrict__ isExerciseDate_global,
										   const float* __restrict__  rates1_global, 
										   const float*  __restrict__ rates2_global, 
										   const float*  __restrict__ rates3_global, 
										   const float*  __restrict__ strategyVariables_global,
										   int strategyVariables_maxSize,
										   const float*  __restrict__ forwards_global, 
										   const float*  __restrict__ discRatios_global
										   ,
										   float* estContinuationValues, // exdates \times paths
										   int* exerciseIndices     // paths
										   )
{
	extern __shared__ float productStrategyData_shared[];

	float* exerciseStrategyDataFloat_s = productStrategyData_shared;

	int tx= threadIdx.x;
	while (tx < exerciseStrategyDataFloat_size)
	{
		exerciseStrategyDataFloat_s[tx] = exerciseStrategyDataFloat_global[tx];
		tx += blockDim.x;
	}

	int* exerciseStrategyDataInt_s = reinterpret_cast<int*>(exerciseStrategyDataFloat_s+exerciseStrategyDataFloat_size);

	tx= threadIdx.x;
	while (tx < exerciseStrategyDataInt_size)
	{
		exerciseStrategyDataInt_s[tx] = exerciseStrategyDataInt_global[tx];
		tx += blockDim.x;
	}

	bool* exerciseIndicators_s = reinterpret_cast<bool*>(exerciseStrategyDataInt_s+exerciseStrategyDataInt_size);


	tx= threadIdx.x;
	while (tx < numberSteps)
	{
		exerciseIndicators_s[tx] = isExerciseDate_global[tx];
		tx += blockDim.x;
	}
	__syncthreads();

	//  We have moved all the auxiliary data into shared memory.
	// It is the same for all threads so sharing is good!

	// create objects, it is up to them what they want to store
	R product(productDataTextureOffset,numberSteps);
	S exerciseValue(exerciseValueFloat_tex_offset, exerciseValueInt_tex_offset,numberSteps);
	T exerciseStrategy(numberSteps,numberExerciseDates,exerciseStrategyDataFloat_s,exerciseStrategyDataInt_s);

	int path0 = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int p = path0; p < paths; p+=stride)
	{
		product.newPath();

		bool done=false;

		int i=0; // maintain i in scope 
		int exerciseIndex=0;

		for (; i < numberSteps && !done; ++i)        
		{

			float flow1=0.0;        
			float flow2=0.0;
			float rate1 = rates1_global[i*paths+p];
			float rate2 = rates2_global[i*paths+p];
			float rate3 = rates3_global[i*paths+p];

			bool exercise=false;

			if (exerciseIndicators_s[i])
			{
				float exValue= exerciseValue.exerciseValue(i,exerciseIndex,rate1,rate2,rate3,p,paths,forwards_global,discRatios_global);
				float estContValue=-1.0E20;


				exercise = exerciseStrategy(paths,
					numberSteps,
					numberExerciseDates,
					exerciseStrategyDataFloat_s,
					exerciseStrategyDataInt_s,
					exValue,
					p,
					exerciseIndex,
					strategyVariables_global,
					strategyVariables_maxSize,
					rate1,rate2,rate3,forwards_global,discRatios_global, estContValue
					);

				estContinuationValues[i*paths+p] = estContValue;


				if (exercise)
				{
					flow1 = exValue;
					done = true;

				}

				++exerciseIndex;
			}

			if (!exercise)
				done = product.getCashFlows(flow1,flow2,rate1,rate2,rate3,forwards_global,discRatios_global,paths);

			int location = i*paths+p;

			genFlows1_global[location] = flow1;
			genFlows2_global[location] = flow2;

		}  

		exerciseIndices[p] = i;

		for (; i < numberSteps; ++i)        
		{
			int location = i*paths+p;
			genFlows1_global[location] = 0.0f;
			genFlows2_global[location] = 0.0f;

		}

	}

}


template<class R, class S, class T>
__global__ void cashFlowGeneratorEE_unsharing_kernel(float* genFlows1_global,
													 float*  genFlows2_global, 
													 int productDataTextureOffset,
													 int exerciseValueFloat_tex_offset,
													 int exerciseValueInt_tex_offset,
													 float*  exerciseStrategyDataFloat_global,
													 int exerciseStrategyDataFloat_size,
													 int*  exerciseStrategyDataInt_global,
													 int  exerciseStrategyDataInt_size,
													 int paths, 
													 int numberSteps,
													 int numberExerciseDates, 
													 bool* isExerciseDate_global,
													 float*  rates1_global, 
													 float*  rates2_global, 
													 float*  rates3_global, 
													 float*  strategyVariables_global,
													 int strategyVariables_maxSize,
													 float*  forwards_global, 
													 float*  discRatios_global
													 ,
													 float* estContinuationValues, // exdates \times paths
													 int* exerciseIndices     // paths
													 )
{
	extern __shared__ float productStrategyData_shared[];

	//	float* exerciseStrategyDataFloat_s = productStrategyData_shared;

	//	int tx= threadIdx.x;
	//	while (tx < exerciseStrategyDataFloat_size)
	//	{
	//		exerciseStrategyDataFloat_s[tx] = exerciseStrategyDataFloat_global[tx];
	//		tx += blockDim.x;
	//	}

	int* exerciseStrategyDataInt_s = reinterpret_cast<int*>(productStrategyData_shared);
	//exerciseStrategyDataFloat_s+exerciseStrategyDataFloat_size);

	int tx= threadIdx.x;
	while (tx < exerciseStrategyDataInt_size)
	{
		exerciseStrategyDataInt_s[tx] = exerciseStrategyDataInt_global[tx];
		tx += blockDim.x;
	}

	bool* exerciseIndicators_s = reinterpret_cast<bool*>(exerciseStrategyDataInt_s+exerciseStrategyDataInt_size);


	tx= threadIdx.x;
	while (tx < numberSteps)
	{
		exerciseIndicators_s[tx] = isExerciseDate_global[tx];
		tx += blockDim.x;
	}
	__syncthreads();

	//  We have moved some of  the auxiliary data into shared memory.
	// It is the same for all threads so sharing is good!

	// create objects, it is up to them what they want to store
	R product(productDataTextureOffset,numberSteps);
	S exerciseValue(exerciseValueFloat_tex_offset, exerciseValueInt_tex_offset,numberSteps);
	T exerciseStrategy(numberSteps,numberExerciseDates,exerciseStrategyDataFloat_global,exerciseStrategyDataInt_s);

	int path0 = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;

	for (int p = path0; p < paths; p+=stride)
	{
		product.newPath();

		bool done=false;

		int i=0; // maintain i in scope 
		int exerciseIndex=0;

		for (; i < numberSteps && !done; ++i)        
		{

			float flow1=0.0;        
			float flow2=0.0;
			float rate1 = rates1_global[i*paths+p];
			float rate2 = rates2_global[i*paths+p];
			float rate3 = rates3_global[i*paths+p];

			bool exercise=false;

			if (exerciseIndicators_s[i])
			{
				float exValue= exerciseValue.exerciseValue(i,exerciseIndex,rate1,rate2,rate3,p,paths,forwards_global,discRatios_global);
				float estContValue=-1.0E20;


				exercise = exerciseStrategy(paths,
					numberSteps,
					numberExerciseDates,
					exerciseStrategyDataFloat_global,
					exerciseStrategyDataInt_s,
					exValue,
					p,
					exerciseIndex,
					strategyVariables_global,
					strategyVariables_maxSize,
					rate1,rate2,rate3,forwards_global,discRatios_global, estContValue
					);

				estContinuationValues[exerciseIndex*paths+p] = estContValue;


				if (exercise)
				{
					flow1 = exValue;
					done = true;

				}

				++exerciseIndex;
			}

			if (!exercise)
				done = product.getCashFlows(flow1,flow2,rate1,rate2,rate3,forwards_global,discRatios_global,paths);

			int location = i*paths+p;

			genFlows1_global[location] = flow1;
			genFlows2_global[location] = flow2;

		}  

		exerciseIndices[p] = i;

		for (; i < numberSteps; ++i)        
		{
			int location = i*paths+p;
			genFlows1_global[location] = 0.0f;
			genFlows2_global[location] = 0.0f;

		}

	}

}






template<class R,class S,class T>
void cashFlowGeneratorEE_gpu(
	float*  genFlows1_global,
	float*  genFlows2_global, 
	float*  productData_global,
	float*  exerciseValueDataFloat_global,
	int*    exerciseValueDataInt_global,
	float*  exerciseStrategyDataFloat_global,
	int exerciseStrategyDataFloat_size,
	int*  exerciseStrategyDataInt_global,
	int exerciseStrategyDataInt_size,
	int paths, 
	int numberSteps,
	int numberExerciseDates,
	bool* isExerciseDate_global,
	float*  rates1_global, 
	float*  rates2_global, 
	float*  rates3_global, 
	float*  strategyVariables_global,
	int strategyVariables_maxSize,
	float*  forwards_global, 
	float*  discRatios_global
	, float* estContinuationValues, // exdates \times paths
	int* exerciseIndices    
	)   
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

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	// set texture parameters for product
	tex_aux_data.addressMode[0] = cudaAddressModeWrap;
	tex_aux_data.addressMode[1] = cudaAddressModeWrap;
	tex_aux_data.filterMode = cudaFilterModeLinear;
	tex_aux_data.normalized = false;    // access with normalized texture coordinates
	cudaBindTexture( NULL, tex_aux_data, productData_global, channelDesc);


	// set texture parameters for exercise value
	tex_exvalue_aux_float_data.addressMode[0] = cudaAddressModeWrap;
	tex_exvalue_aux_float_data.addressMode[1] = cudaAddressModeWrap;
	tex_exvalue_aux_float_data.filterMode = cudaFilterModeLinear;
	tex_exvalue_aux_float_data.normalized = false;    // access with normalized texture coordinates
	cudaBindTexture( NULL, tex_exvalue_aux_float_data, exerciseValueDataFloat_global, channelDesc);

	cudaBindTexture( NULL, tex_exvalue_aux_int_data, exerciseValueDataInt_global);

	int size1
		= numberSteps*sizeof(bool)+exerciseStrategyDataInt_size*sizeof(int)+exerciseStrategyDataFloat_size*sizeof(float);

	int size2
		= numberSteps*sizeof(bool)+exerciseStrategyDataInt_size*sizeof(int);

	int device;
	cudaDeviceProp deviceproperty;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&deviceproperty, device);          



	int productDataTextureOffset=0;
	int exerciseValueFloat_tex_offset=0;
	int exerciseValueInt_tex_offset=0;

	bool avoidShared = true;

	  CUT_CHECK_ERR("cashFlowGeneratorEE_kernel pre-failure \n");

	if ((static_cast<int>(deviceproperty.sharedMemPerBlock) >= size1) && !avoidShared)
	{
        
        ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);
//		std::cout << "early ex cash flow generation using shared\n";
		cashFlowGeneratorEE_kernel<R,S,T><<<dimGrid, dimBlock,size1>>>(genFlows1_global,
			genFlows2_global, 
			productDataTextureOffset,
			exerciseValueFloat_tex_offset,
			exerciseValueInt_tex_offset,
			exerciseStrategyDataFloat_global,
			exerciseStrategyDataFloat_size,
			exerciseStrategyDataInt_global,
			exerciseStrategyDataInt_size,
			paths, 
			numberSteps,
			numberExerciseDates, 
			isExerciseDate_global, // number of trues should equal numberExerciseDates
			rates1_global, 
			rates2_global, 
			rates3_global, 
			strategyVariables_global,
			strategyVariables_maxSize,
			forwards_global, 
			discRatios_global
			,
			estContinuationValues, // exdates \times paths
			exerciseIndices     
			);
	}
	else
	{
		//std::cout << "early ex cash flow generation not using shared\n";
                
        ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);
		cashFlowGeneratorEE_unsharing_kernel<R,S,T><<<dimGrid, dimBlock,size2>>>(genFlows1_global,
			genFlows2_global, 
			productDataTextureOffset,
			exerciseValueFloat_tex_offset,
			exerciseValueInt_tex_offset,
			exerciseStrategyDataFloat_global,
			exerciseStrategyDataFloat_size,
			exerciseStrategyDataInt_global,
			exerciseStrategyDataInt_size,
			paths, 
			numberSteps,
			numberExerciseDates, 
			isExerciseDate_global, // number of trues should equal numberExerciseDates
			rates1_global, 
			rates2_global, 
			rates3_global, 
			strategyVariables_global,
			strategyVariables_maxSize,
			forwards_global, 
			discRatios_global
			,
			estContinuationValues, // exdates \times paths
			exerciseIndices     
			);
	}
		cudaThreadSynchronize();
	CUT_CHECK_ERR("cashFlowGeneratorEE_kernel post-failure \n");


	cudaUnbindTexture(tex_exvalue_aux_int_data);
	cudaUnbindTexture(tex_exvalue_aux_float_data);
	cudaUnbindTexture(tex_aux_data);

		CUT_CHECK_ERR("cashFlowGeneratorEE_kernel post-texture failure \n");


}





class LSAExerciseStrategyQuadratic_gpu
{
public:

	__host__   static void outputDataVectorSize(int& floatDataSize,
		int& intDataSize, 
		int numberExerciseTimes,
		const std::vector<int>& variablesPerStep)
	{
		intDataSize= numberExerciseTimes+1;
		floatDataSize=(2*(*std::max_element(variablesPerStep.begin(),variablesPerStep.end()))+1)*numberExerciseTimes // weights for LS
			+ numberExerciseTimes ; // basis shifts                 
	}




	template<class S, class T,class U, class V>
	__host__   
		static void outputDataVectors(S floatDataIterator, 
		T intDataIterator,
		int numberExerciseTimes,
		const std::vector<int>& variablesPerStep,
		const std::vector<U>& basisShifts,
		const std::vector<V>& basisWeights)
	{

		int v =  *std::max_element(variablesPerStep.begin(),variablesPerStep.end());
		int rowSize = 2*v+1;
		*intDataIterator = rowSize;
		++intDataIterator;
		//    int wSize = (2*v+1)*numberExerciseTimes;

		for (int i=0; i < numberExerciseTimes; ++i, ++floatDataIterator, ++intDataIterator)
		{
			*floatDataIterator = static_cast<float>(basisShifts[i]);
			*intDataIterator = variablesPerStep[i];

		}

		for (int i=0; i < numberExerciseTimes; ++i)
		{
			for (int j=0; j < rowSize; ++j, ++floatDataIterator)
				*floatDataIterator = static_cast<float>(basisWeights[i*rowSize+j]);
		}



	}


	__device__       
		LSAExerciseStrategyQuadratic_gpu(int steps, int numberExerciseTimes, float* strat_float_data,
		int* strat_int_dat)
	{}



	__device__       
		bool operator()(int paths,
		int steps, 
		int numberExerciseTimes,
		float* strat_float_data,
		int* strat_int_data,
		float exValue, 
		int p,
		int exerciseIndex,
		float* strategyVariables,
		int strategyVariables_maxSize,
		float rate1,
		float rate2,
		float rate3,
		float* v,
		float* w
			,
			float& estCValue
		) const
	{
		int numberVariables = strat_int_data[exerciseIndex+1];
		//	int numberDataPoints = 2*numberVariables+1;
		int rowSize = strat_int_data[0];

		float value = strat_float_data[exerciseIndex];

		float* weightPtr = strat_float_data+numberExerciseTimes+rowSize*exerciseIndex;

		value += *weightPtr;
		++weightPtr;

		for (int i=0; i < numberVariables; ++i, ++weightPtr)
		{
			float v = strategyVariables[p+i*paths+exerciseIndex*paths*strategyVariables_maxSize];
			value += (*weightPtr)*v;
			++weightPtr;
			value += (*weightPtr)*v*v;
		}

		estCValue = value;

		return value < exValue;

	}


private:


};


class LSAExerciseStrategyQuadraticCross_gpu
{
public:

	__host__   static void outputDataVectorSize(int& floatDataSize,
		int& intDataSize, 
		int numberExerciseTimes,
		const std::vector<int>& variablesPerStep)
	{
		intDataSize= numberExerciseTimes+1;

		// you may need less if only one or two variables but this is a comfortable upper bound
		floatDataSize=(3*(*std::max_element(variablesPerStep.begin(),variablesPerStep.end()))+1)*numberExerciseTimes // weights for LS
			+ numberExerciseTimes ; // basis shifts                 
	}




	template<class S, class T,class U, class V>
	__host__   
		static void outputDataVectors(S floatDataIterator, 
		T intDataIterator,
		int numberExerciseTimes,
		const std::vector<int>& variablesPerStep,
		const std::vector<U>& basisShifts,
		const std::vector<V>& basisWeights)
	{

		int v =  *std::max_element(variablesPerStep.begin(),variablesPerStep.end());
		int rowSize = 3*v+1;

		if (v==1)
			rowSize = 3;
		if (v==2)
			rowSize = 6;

		*intDataIterator = rowSize;
		++intDataIterator;

		for (int i=0; i < numberExerciseTimes; ++i, ++floatDataIterator, ++intDataIterator)
		{
			*floatDataIterator = static_cast<float>(basisShifts[i]);
			*intDataIterator = variablesPerStep[i];

		}

		for (int i=0; i < numberExerciseTimes; ++i)
		{
			for (int j=0; j < rowSize; ++j, ++floatDataIterator)
				*floatDataIterator = static_cast<float>(basisWeights[i*rowSize+j]);
		}



	}


	__device__       
		LSAExerciseStrategyQuadraticCross_gpu(int steps, int numberExerciseTimes, float* strat_float_data,
		int* strat_int_dat)
	{}



	__device__       
		bool operator()(int paths,
		int steps, 
		int numberExerciseTimes,
		float* strat_float_data,
		int* strat_int_data,
		float exValue, 
		int p,
		int exerciseIndex,
		float* strategyVariables,
		int strategyVariables_maxSize,
		float rate1,
		float rate2,
		float rate3,
		const float* __restrict__ v,
		const float* __restrict__ w
		,
		float& value
		) const
	{
		int numberVariables = strat_int_data[exerciseIndex+1];
		int rowSize = strat_int_data[0];

		//float
		value = strat_float_data[exerciseIndex];

		float* weightPtr = strat_float_data+numberExerciseTimes+rowSize*exerciseIndex;

		value += *weightPtr;
		++weightPtr;

		if (numberVariables ==1)
		{
			float v = strategyVariables[p+exerciseIndex*paths*strategyVariables_maxSize];
			value += (*weightPtr)*v;
			++weightPtr;
			value += (*weightPtr)*v*v;
			return  value < exValue;
		}

		if (numberVariables ==2)
		{
			float x = strategyVariables[p+exerciseIndex*paths*strategyVariables_maxSize];
			float y = strategyVariables[p+paths+exerciseIndex*paths*strategyVariables_maxSize];

			value += (*weightPtr)*x;
			++weightPtr;
			value += (*weightPtr)*y;
			++weightPtr;
			value += (*weightPtr)*x*x;
			++weightPtr;
			value += (*weightPtr)*x*y;
			++weightPtr;
			value += (*weightPtr)*y*y;
			++weightPtr;


			return  value < exValue;
		}

		// general case remains

		// first do linear terms
		for (int i=0; i < numberVariables; ++i, ++weightPtr)
		{
			float v = strategyVariables[p+i*paths+exerciseIndex*paths*strategyVariables_maxSize];
			value += (*weightPtr)*v;
		}

		// now do cross terms
		for (int i=0; i < numberVariables; ++i, ++weightPtr)
		{
			float x = strategyVariables[p+i*paths+exerciseIndex*paths*strategyVariables_maxSize];
			int k = (i+1) % numberVariables;
			float y = strategyVariables[p+k*paths+exerciseIndex*paths*strategyVariables_maxSize];
			value += (*weightPtr)*x*y;
		}

		// now do squares
		for (int i=0; i < numberVariables; ++i, ++weightPtr)
		{
			float v = strategyVariables[p+i*paths+exerciseIndex*paths*strategyVariables_maxSize];
			value += (*weightPtr)*v*v;
		}

		//		estCValue = value;

		return value < exValue;

	}


private:


};


class LSAMultiExerciseStrategyQuadraticCross_gpu
{
public:

	__host__   static void outputDataVectorSize(int& floatDataSize,
		int& intDataSize, 
		int MaxRegressionDepth,
		int numberExerciseTimes,
		const std::vector<int>& variablesPerStep)
	{
		intDataSize= numberExerciseTimes+3;
		int maxVariableSize = *std::max_element(variablesPerStep.begin(),variablesPerStep.end());

		// you may need less if only one or two variables but this is a comfortable upper bound
		floatDataSize=(3*maxVariableSize+1)*numberExerciseTimes*MaxRegressionDepth // weights for LS
			+2*maxVariableSize*numberExerciseTimes*MaxRegressionDepth // means and standard deviations
			+ 3*MaxRegressionDepth*numberExerciseTimes; // basis shifts, lower cuts, upper cuts,

	}
	/*
	int data: MAxREgressionDepth, RowSize, variables per step
	float data: 
	Andersenshifts, numberExerciseTimes
	variableShiftsToSubtract,  numberExerciseTimes*MaxRegressionDepth*maxVariableSize
	variableDivisors           numberExerciseTimes*MaxRegressionDepth*maxVariableSize
	low cutoffs,               numberExerciseTimes*MaxRegressionDepth
	upper cutoffs,             numberExerciseTimes*MaxRegressionDepth
	basisWeights,              numberExerciseTimes*MaxRegressionDepth*RowSize
	*/

	template<class S, class T,class U, class V, class W, class X>
	__host__   
		static void outputDataVectors(S floatDataIterator, 
		T intDataIterator,
		int MaxRegressionDepth,
		int numberExerciseTimes,
		const std::vector<int>& variablesPerStep,
		const std::vector<U>& basisShifts,
		const std::vector<X>& variableShiftsToSubtract,
		const std::vector<X>& variableDivisors,
		const std::vector<W>& lowerCutoffs,
		const std::vector<W>& upperCutoffs,
		const std::vector<V>& basisWeights)
	{
//		S beg = floatDataIterator;
		*intDataIterator = MaxRegressionDepth;
		++intDataIterator;

		int v =  *std::max_element(variablesPerStep.begin(),variablesPerStep.end());
		int rowSize = (3*v+1);

		if (v==1)
			rowSize = 3;
		if (v==2)
			rowSize = 6;

		*intDataIterator = v;
		++intDataIterator;

		*intDataIterator = rowSize;
		++intDataIterator;

		for (int i=0; i < numberExerciseTimes; ++i, ++floatDataIterator, ++intDataIterator)
		{
			*floatDataIterator = static_cast<float>(basisShifts[i]);
			*intDataIterator = variablesPerStep[i];
		}

		for (int i=0, l=0; i < numberExerciseTimes; ++i)
			for (int k=0; k < MaxRegressionDepth; ++k )
				for (int m=0; m < v; ++m, ++floatDataIterator,++l)
					*floatDataIterator = static_cast<float>(variableShiftsToSubtract[l]);

		for (int i=0, l=0; i < numberExerciseTimes; ++i)
			for (int k=0; k < MaxRegressionDepth; ++k)
				for (int m=0; m < v; ++m, ++floatDataIterator,++l)
					*floatDataIterator = static_cast<float>(variableDivisors[l]);

		for (int i=0, l=0; i < numberExerciseTimes; ++i)
			for (int k=0; k < MaxRegressionDepth; ++k,++l, ++floatDataIterator)
				*floatDataIterator = static_cast<float>(lowerCutoffs[l]);

		for (int i=0, l=0; i < numberExerciseTimes; ++i)
			for (int k=0; k < MaxRegressionDepth; ++k,++l, ++floatDataIterator)
				*floatDataIterator = static_cast<float>(upperCutoffs[l]);

		CubeConstFacade<double> basis_Cube(&basisWeights[0],numberExerciseTimes,MaxRegressionDepth,rowSize);

		for (int i=0; i < numberExerciseTimes; ++i)
			for (int k=0; k < MaxRegressionDepth; ++k)
				for (int j=0; j < rowSize; ++j, ++floatDataIterator)
				{
					*floatDataIterator =  static_cast<float>(basis_Cube(i,k,j));
					
				}



	}


	__device__       
		LSAMultiExerciseStrategyQuadraticCross_gpu(int steps, int numberExerciseTimes, const float* __restrict__ strat_float_data,
		const int* __restrict__ strat_int_dat)
	{}



	__device__       
		bool operator()(int paths,
		int steps, 
		int numberExerciseTimes,
		const float* __restrict__ strat_float_data,
		const int* __restrict__ strat_int_data,
		float exValue, 
		int p,
		int exerciseIndex,
		const float* __restrict__ strategyVariables,
		int strategyVariables_maxSize,
		float rate1,
		float rate2,
		float rate3,
		const float* __restrict__ v,
		const float* __restrict__ w
		,
		float& value
		) const
	{

		/*
		int data: MAxREgressionDepth, RowSize, variables per step
		float data: 
		Andersenshifts, numberExerciseTimes
		variableShiftsToSubtract,  numberExerciseTimes*MaxRegressionDepth
		variableDivisors           numberExerciseTimes*MaxRegressionDepth

		low cutoffs, numberExerciseTimes*MaxRegressionDepth
		upper cutoffs,  numberExerciseTimes*MaxRegressionDepth
		basisWeights, numberExerciseTimes*MaxRegressionDepth*RowSize
		*/

		// get integer data
		int regressionDepth = strat_int_data[0];
		int maxVariables = strat_int_data[1];
		int rowSize = strat_int_data[2];
		int numberVariables = strat_int_data[exerciseIndex+3];

		int layerSize = rowSize*regressionDepth;

		//	float value;

		bool stillGoing = true;
		int depth=0; // i.e. current depth

		const float* __restrict__ variableShiftsToSubtractStartG= strat_float_data+numberExerciseTimes+exerciseIndex*regressionDepth*maxVariables;
		const float* __restrict__ variableDivisorsStartG =variableShiftsToSubtractStartG+numberExerciseTimes*regressionDepth*maxVariables;
		const float* __restrict__ lowerCutsStart= strat_float_data+2*numberExerciseTimes*regressionDepth*maxVariables+numberExerciseTimes+exerciseIndex*regressionDepth;
		const float* __restrict__ upperCutsStart= lowerCutsStart+numberExerciseTimes*regressionDepth;


//		float firstWeight;

		while (stillGoing)
		{
			value = strat_float_data[exerciseIndex];

			const float* __restrict__ weightPtr = strat_float_data+
				numberExerciseTimes+
				2*numberExerciseTimes*regressionDepth*maxVariables+
				2*numberExerciseTimes*regressionDepth+
				layerSize*exerciseIndex+depth*rowSize;

//			firstWeight =weightPtr-strat_float_data ;

			value += *weightPtr;
			++weightPtr;

			const float* __restrict__ variableShiftsToSubtractStartD =variableShiftsToSubtractStartG + maxVariables*depth;
			const float* __restrict__ variableDivisorsStartD =variableDivisorsStartG + maxVariables*depth;

			if (numberVariables ==1)
			{
				float v = strategyVariables[p+exerciseIndex*paths*strategyVariables_maxSize];
				v -= variableShiftsToSubtractStartD[0];
				v /= variableDivisorsStartD[0];
				value += (*weightPtr)*v;
				++weightPtr;
				value += (*weightPtr)*v*v;
			} 
			else
				if (numberVariables ==2)
				{
					float x = strategyVariables[p+exerciseIndex*paths*strategyVariables_maxSize];
					x -= variableShiftsToSubtractStartD[0];
					x /= variableDivisorsStartD[0];

					float y = strategyVariables[p+paths+exerciseIndex*paths*strategyVariables_maxSize];
					y -= variableShiftsToSubtractStartD[1];
					y /= variableDivisorsStartD[1];

					value += (*weightPtr)*x;
					++weightPtr;
					value += (*weightPtr)*y;
					++weightPtr;
					value += (*weightPtr)*x*x;
					++weightPtr;
					value += (*weightPtr)*x*y;
					++weightPtr;
					value += (*weightPtr)*y*y;
					++weightPtr;
				}
				else
				{

					// general case remains

					// first do linear terms
					for (int i=0; i < numberVariables; ++i, ++weightPtr)
					{
						float v = strategyVariables[p+i*paths+exerciseIndex*paths*strategyVariables_maxSize];
						v -= variableShiftsToSubtractStartD[i];
						v /= variableDivisorsStartD[i];

						value += (*weightPtr)*v;
					}

					// now do cross terms
					for (int i=0; i < numberVariables; ++i, ++weightPtr)
					{
						float x = strategyVariables[p+i*paths+exerciseIndex*paths*strategyVariables_maxSize];
						x -= variableShiftsToSubtractStartD[i];
						x /= variableDivisorsStartD[i];

						int k = (i+1) % numberVariables;
						float y = strategyVariables[p+k*paths+exerciseIndex*paths*strategyVariables_maxSize];
						y -= variableShiftsToSubtractStartD[k];
						y /= variableDivisorsStartD[k];

						value += (*weightPtr)*x*y;
					}

					// now do squares
					for (int i=0; i < numberVariables; ++i, ++weightPtr)
					{
						float v = strategyVariables[p+i*paths+exerciseIndex*paths*strategyVariables_maxSize];
						v -= variableShiftsToSubtractStartD[i];
						v /= variableDivisorsStartD[i];

						value += (*weightPtr)*v*v;
					}
				}

				++depth;
				if (depth == regressionDepth)
				{
					stillGoing=false;

				}
				else
				{

					float lowerCutValue = lowerCutsStart[depth-1];

					if (value-exValue < lowerCutValue)
					{
						value = lowerCutValue;
						return true;
					}
					float upperCutValue = upperCutsStart[depth-1];

					if (value-exValue >upperCutValue)
					{
						value = upperCutValue;
						return false;
					}
				}

		}

		bool exercise = value < exValue;

		return exercise;


	}


private:


};



class LSAMultiExerciseStrategyQuadratic_gpu
{
public:

	__host__   static void outputDataVectorSize(int& floatDataSize,
		int& intDataSize, 
		int MaxRegressionDepth,
		int numberExerciseTimes,
		const std::vector<int>& variablesPerStep)
	{
		intDataSize= numberExerciseTimes+3;
		int maxVariableSize = *std::max_element(variablesPerStep.begin(),variablesPerStep.end());

		// you may need less if only one or two variables but this is a comfortable upper bound
		floatDataSize=(2*maxVariableSize+1)*numberExerciseTimes*MaxRegressionDepth // weights for LS
			+2*maxVariableSize*numberExerciseTimes*MaxRegressionDepth // means and standard deviations
			+ 3*MaxRegressionDepth*numberExerciseTimes; // basis shifts, lower cuts, upper cuts,

	}
	/*
	int data: MAxREgressionDepth, RowSize, variables per step
	float data: 
	Andersenshifts, numberExerciseTimes
	variableShiftsToSubtract,  numberExerciseTimes*MaxRegressionDepth*maxVariableSize
	variableDivisors           numberExerciseTimes*MaxRegressionDepth*maxVariableSize
	low cutoffs,               numberExerciseTimes*MaxRegressionDepth
	upper cutoffs,             numberExerciseTimes*MaxRegressionDepth
	basisWeights,              numberExerciseTimes*MaxRegressionDepth*RowSize
	*/

	template<class S, class T,class U, class V, class W, class X>
	__host__   
		static void outputDataVectors(S floatDataIterator, 
		T intDataIterator,
		int MaxRegressionDepth,
		int numberExerciseTimes,
		const std::vector<int>& variablesPerStep,
		const std::vector<U>& basisShifts,
		const std::vector<X>& variableShiftsToSubtract,
		const std::vector<X>& variableDivisors,
		const std::vector<W>& lowerCutoffs,
		const std::vector<W>& upperCutoffs,
		const std::vector<V>& basisWeights)
	{
	//	S beg = floatDataIterator;
		*intDataIterator = MaxRegressionDepth;
		++intDataIterator;

		int v =  *std::max_element(variablesPerStep.begin(),variablesPerStep.end());
		int rowSize = (2*v+1);

		*intDataIterator = v;
		++intDataIterator;

		*intDataIterator = rowSize;
		++intDataIterator;

		for (int i=0; i < numberExerciseTimes; ++i, ++floatDataIterator, ++intDataIterator)
		{
			*floatDataIterator = static_cast<float>(basisShifts[i]);
			*intDataIterator = variablesPerStep[i];
		}

		for (int i=0, l=0; i < numberExerciseTimes; ++i)
			for (int k=0; k < MaxRegressionDepth; ++k )
				for (int m=0; m < v; ++m, ++floatDataIterator,++l)
					*floatDataIterator = static_cast<float>(variableShiftsToSubtract[l]);

		for (int i=0, l=0; i < numberExerciseTimes; ++i)
			for (int k=0; k < MaxRegressionDepth; ++k)
				for (int m=0; m < v; ++m, ++floatDataIterator,++l)
					*floatDataIterator = static_cast<float>(variableDivisors[l]);

		for (int i=0, l=0; i < numberExerciseTimes; ++i)
			for (int k=0; k < MaxRegressionDepth; ++k,++l, ++floatDataIterator)
				*floatDataIterator = static_cast<float>(lowerCutoffs[l]);

		for (int i=0, l=0; i < numberExerciseTimes; ++i)
			for (int k=0; k < MaxRegressionDepth; ++k,++l, ++floatDataIterator)
				*floatDataIterator = static_cast<float>(upperCutoffs[l]);

		CubeConstFacade<double> basis_Cube(&basisWeights[0],numberExerciseTimes,MaxRegressionDepth,rowSize);

		for (int i=0; i < numberExerciseTimes; ++i)
			for (int k=0; k < MaxRegressionDepth; ++k)
				for (int j=0; j < rowSize; ++j, ++floatDataIterator)
				{
				//	float val =;
				//	int l = floatDataIterator-beg;
					*floatDataIterator =   static_cast<float>(basis_Cube(i,k,j));
				}



	}


	__device__       
		LSAMultiExerciseStrategyQuadratic_gpu(int steps, int numberExerciseTimes, const float* __restrict__ strat_float_data,
		const int* __restrict__  strat_int_dat)
	{}



	__device__       
		bool operator()(int paths,
		int steps, 
		int numberExerciseTimes,
		const float* __restrict__ strat_float_data,
		const int* __restrict__  strat_int_data,
		float exValue, 
		int p,
		int exerciseIndex,
		const float* __restrict__ strategyVariables,
		int strategyVariables_maxSize,
		float rate1,
		float rate2,
		float rate3,
		const float* __restrict__ v,
		const float* __restrict__ w
		,
		float& value
		) const
	{

		/*
		int data: MAxREgressionDepth, RowSize, variables per step
		float data: 
		Andersenshifts, numberExerciseTimes
		variableShiftsToSubtract,  numberExerciseTimes*MaxRegressionDepth
		variableDivisors           numberExerciseTimes*MaxRegressionDepth

		low cutoffs, numberExerciseTimes*MaxRegressionDepth
		upper cutoffs,  numberExerciseTimes*MaxRegressionDepth
		basisWeights, numberExerciseTimes*MaxRegressionDepth*RowSize
		*/

		// get integer data
		int regressionDepth = strat_int_data[0];
		int maxVariables = strat_int_data[1];
		int rowSize = strat_int_data[2];
		int numberVariables = strat_int_data[exerciseIndex+3];

		int layerSize = rowSize*regressionDepth;

		//	float value;

		bool stillGoing = true;
		int depth=0; // i.e. current depth

		const float* __restrict__ variableShiftsToSubtractStartG= strat_float_data+numberExerciseTimes+exerciseIndex*regressionDepth*maxVariables;
		const float* __restrict__ variableDivisorsStartG =variableShiftsToSubtractStartG+numberExerciseTimes*regressionDepth*maxVariables;
		const float* __restrict__ lowerCutsStart= strat_float_data+2*numberExerciseTimes*regressionDepth*maxVariables+numberExerciseTimes+exerciseIndex*regressionDepth;
		const float* __restrict__ upperCutsStart= lowerCutsStart+numberExerciseTimes*regressionDepth;


//		float firstWeight;

		while (stillGoing)
		{
			value = strat_float_data[exerciseIndex];

			const float* __restrict__ weightPtr = strat_float_data+
				numberExerciseTimes+
				2*numberExerciseTimes*regressionDepth*maxVariables+
				2*numberExerciseTimes*regressionDepth+
				layerSize*exerciseIndex+depth*rowSize;

//			firstWeight =weightPtr-strat_float_data ;

			value += *weightPtr;
			++weightPtr;

			const float* __restrict__ variableShiftsToSubtractStartD =variableShiftsToSubtractStartG + maxVariables*depth;
			const float* __restrict__ variableDivisorsStartD =variableDivisorsStartG + maxVariables*depth;



			// general case remains

			// first do linear terms
			for (int i=0; i < numberVariables; ++i, ++weightPtr)
			{
				float v = strategyVariables[p+i*paths+exerciseIndex*paths*strategyVariables_maxSize];
				v -= variableShiftsToSubtractStartD[i];
				v /= variableDivisorsStartD[i];

				value += (*weightPtr)*v;
				++weightPtr;
				value += (*weightPtr)*v*v;
			}


			++depth;
			if (depth == regressionDepth)
			{
				stillGoing=false;

			}
			else
			{

				float lowerCutValue = lowerCutsStart[depth-1];

				if (value-exValue < lowerCutValue)
				{
					value = lowerCutValue;
					return true;
				}
				float upperCutValue = upperCutsStart[depth-1];

				if (value-exValue >upperCutValue)
				{
					value = upperCutValue;
					return false;
				}
			}

		}

		bool exercise = value < exValue;

		return exercise;


	}


private:


};



#endif

