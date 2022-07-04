//
//
//							basis Functions.cu
//
//

// could use two textures tex_aux_float_data
//  tex_aux_int_data

#include "basisFunctions.h"
#include <cudaWrappers/cudaTextureBinder.h>
#include <gpuCompatibilityCheck.h>

#define SHARED_BUFFER_SIZE 2048
#define SHARED_DATA_PER_THREAD 8

texture<float, 1, cudaReadModeElementType> tex_aux_float_data;
texture<int, 1> tex_aux_int_data;



// not path-dependent
class simpleBasisVariables
{
private:
	// no data members
public:

	__device__  simpleBasisVariables(int offsetIntegerData,
		int offsetFloatData,
		int numberRates)
	{

	}

	__host__  simpleBasisVariables(int* integerData,
		float* floatData,
		int numberRates)
	{

	}

	__device__ __host__ int numberOfVariables(int step, int numberRates)
	{
		if (step+1 < numberRates)
			return 3;
		else 
			return 2;
	}

	__device__ void getVariableValues(float* sharedMemoryOffset, 
		int offsetIntegerData,
		int offsetFloatData,
		int numberOfRatesReset, 
		int numberRates, 
		float rate1, 
		float rate2, 
		float rate3, 
		float* allForwardRates, 
		float* allDiscountRatios,
		int pathNumber,
		int paths
		)
	{
		sharedMemoryOffset[0]=rate1;
		if (numberOfRatesReset+1 < numberRates)
		{  
			sharedMemoryOffset[1]=rate2;
			sharedMemoryOffset[2]=allDiscountRatios[(numberRates-1)*numberRates*paths+ paths*numberOfRatesReset+pathNumber];
		}
		else
			sharedMemoryOffset[1] = allDiscountRatios[(numberRates-1)*numberRates*paths+ paths*numberOfRatesReset+pathNumber];

	}


};

// uses texture tex_aux_float_data

template<class T>
class quadraticBasisValues
{
private:
	T basisVariables_;

public:

	__device__ quadraticBasisValues(int offsetIntegerData,
		int offsetFloatData,
		int numberRates) : basisVariables_(offsetIntegerData,offsetFloatData,numberRates)
	{
	}


	__device__ float getImpliedValues(int offsetIntegerData,
		int offsetFloatData,
		int numberOfRates,
		int numberOfRatesReset,
		int step,
		float* sharedWorkingArea,
		float rate1, 
		float rate2, 
		float rate3, 
		float* allForwardRates, 
		float* allDiscountRatios,
		int pathNumber,
		int paths )
	{
		int number = basisVariables_.numberOfVariables(step,numberOfRates);

		basisVariables_.getVariableValues(sharedWorkingArea, 
			offsetIntegerData,
			offsetFloatData,
			numberOfRatesReset, 
			numberOfRates,
			rate1, 
			rate2, 
			rate3, 
			allForwardRates, 
			allDiscountRatios,
			pathNumber,
			paths // to access correct forward rate data 
			);

		float res =  tex1Dfetch(tex_aux_float_data,offsetFloatData);
		++offsetFloatData;
		int i=0;
		for (; i < number; ++i)
		{
			res +=  sharedWorkingArea[i]* tex1Dfetch(tex_aux_float_data,offsetFloatData);
			++offsetFloatData;
		}
		++offsetFloatData;
		for (int k=0; k < number; ++k)
			for (int j=0; j <=k ; ++j, ++i, ++offsetFloatData )
				res +=  sharedWorkingArea[i]* tex1Dfetch(tex_aux_float_data,offsetFloatData);

		return res;

	}



};


template<class T>
class ExerciseStrategyLSA
{
private:

	T basisFunctions_;
	int numberOfSteps_;

public:
	__device__	ExerciseStrategyLSA(int tex_float_aux_offset,
		int tex_int_aux_offset,
		int numberOfSteps)
		:
	basisFunctions_( tex_float_aux_offset+numberOfSteps,
		tex_int_aux_offset+numberOfSteps,
		numberOfSteps), // note this is a requirement on the interface of T
		numberOfSteps_(numberOfSteps)
	{}



	bool __device__ exerciseNow(int step,
		float exValue,
		int tex_float_aux_offset,
		int tex_int_aux_offset,
		int numberOfRatesReset,
		int numberOfRates, 
		float* sharedWorkingArea,
		float rate1, 
		float rate2, 
		float rate3, 
		float* allForwardRates, 
		float* allDiscountRatios,
		int pathNumber,
		int paths 
		)
	{
		int isExerciseTime = tex1Dfetch(tex_aux_int_data,tex_int_aux_offset+numberOfSteps_+step);

		if (isExerciseTime ==0)
			return false;

		int dataOffset = tex1Dfetch(tex_aux_int_data,tex_int_aux_offset+step);



		float estimatedContinuation;
		estimatedContinuation= basisFunctions_.getImpliedValues(tex_int_aux_offset+2*numberOfRates,
			tex_float_aux_offset+dataOffset,
			numberOfRates,
			numberOfRatesReset,
			step, 
			sharedWorkingArea,
			rate1, 
			rate2, 
			rate3, 
			allForwardRates, 
			allDiscountRatios,
			pathNumber,
			paths );



		estimatedContinuation +=  tex1Dfetch(tex_aux_float_data,tex_float_aux_offset+step); // The A part of LSA

		return estimatedContinuation < exValue;


	}

};

//////////////////////////

// kernel
//
// all values are deflated
// genFlow index time step j is value that occurs if we don't break at j. 
// rebate is value that does
// genFlows is paths in lowest dim, followed by steps
// rebate is paths in lowest dim, followed by steps
// all the auxiliary data for the exercise strategy should be in the textures. 

template<class T>
__global__ void exerciseDetermination_kernel(int* stoppingTimes, 
											 float* rebates, 
											 int tex_float_aux_offset,
											 int tex_int_aux_offset,
											 int paths, 
											 int numberSteps,     // implicit assumption that we evolve only to rate times                             
											 float* rates1, 
											 float* rates2, 
											 float* rates3, 
											 float* forwards, 
											 float* discRatios)
{
	__shared__ float sharedWorkingArea[SHARED_BUFFER_SIZE]; // to be fixed
	int bx = blockIdx.x;
	int tx =  threadIdx.x;

	float* threadSharedWorkingArea = sharedWorkingArea+tx*SHARED_DATA_PER_THREAD;

	int gx = gridDim.x;
	int bwidth = blockDim.x;
	int width = gx*bwidth;

	int pathsPerThread = 1 + (( paths -1)/width);
	T exStrategy(tex_float_aux_offset,tex_int_aux_offset,numberSteps);

	for (int l=0; l < pathsPerThread; ++l)
	{
		int pathNumber = width*l + bwidth*bx+tx;
		int dataOffset = pathNumber;
		bool done = false;

		if (pathNumber < paths)
		{
			int stopTime = numberSteps;
			for (int i =0; i < numberSteps && !done; ++i)
			{

				float rate1= rates1[dataOffset];
				float rate2= rates2[dataOffset];
				float rate3= rates3[dataOffset];
				float exValue = rebates[dataOffset];

				done = exStrategy.exerciseNow(i,
					exValue,
					tex_float_aux_offset,
					tex_int_aux_offset,
					i,
					numberSteps, 
					threadSharedWorkingArea,
					rate1,
					rate2,
					rate3,
					forwards,
					discRatios,
					pathNumber,
					paths);

				if (done)
					stopTime =i;
				dataOffset += paths;
			}  // for loop 

			// now store the exercise time
			stoppingTimes[pathNumber] = stopTime;

		} // if pathnumber valid
	}// for loop paths Per Thread
} // end of function




extern "C"
void exerciseTimeDeterminationUsingLSAQuadraticSimpleBasis(int* stoppingTimes_dev_ptr, 
							   float* rebates_dev_ptr, 
							   int* LSAIntData_dev_ptr,
							   float* LSAFloatData_dev_ptr,
							   int paths, 
							   int numberSteps,                                  
							   float* rates1_dev_ptr, 
							   float* rates2_dev_ptr, 
							   float* rates3_dev_ptr, 
							   float* forwards_dev_ptr, 
							   float* discRatios_dev_ptr) // output
{


	const int threadsperblock = 64; // shared memory will break if we use more than 64 threads per block
	const int maxBlocks = 65535;

	// Set up the execution configuration
	dim3 dimGrid;
	dim3 dimBlock;

	dimGrid.x = 1+(paths-1)/threadsperblock;

	if (dimGrid.x > maxBlocks) 
		dimGrid.x=maxBlocks;

	// Fix the number of threads
	dimBlock.x = threadsperblock;

	int tex_float_aux_offset=0;
	int tex_int_aux_offset=0;

	cudaTextureFloatBinder floatBinder(  tex_aux_float_data, LSAFloatData_dev_ptr);
    
	cudaTextureIntBinder intBinder(  tex_aux_int_data, LSAIntData_dev_ptr);
  
     ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);


	exerciseDetermination_kernel<ExerciseStrategyLSA<quadraticBasisValues<simpleBasisVariables > > ><<<dimGrid ,
		dimBlock >>>( stoppingTimes_dev_ptr, 
		rebates_dev_ptr, 
		tex_float_aux_offset,
		tex_int_aux_offset,
		paths, 
		numberSteps,                                  
		rates1_dev_ptr, 
		rates2_dev_ptr, 
		rates3_dev_ptr, 
		forwards_dev_ptr, 
		discRatios_dev_ptr);


}
