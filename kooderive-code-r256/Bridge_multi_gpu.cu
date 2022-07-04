// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3


/*
Multi-dim Brownian bridge example

*/

#include <gold/Bridge_gold.h>
#include "cudaMacros.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cutil.h>

#include <cutil_inline.h>

#include <cudaWrappers/cudaTextureBinder.h>
#include <gpuCompatibilityCheck.h>

#define MAX_BRIDGE_SIZE 256
#define MAX_FACTORS 10

__device__ __constant__ int dev_const_reorderArrayDim[MAX_BRIDGE_SIZE*MAX_FACTORS];
__device__ __constant__ int dev_const_reorderArrayFactor[MAX_BRIDGE_SIZE*MAX_FACTORS];


texture<float, 1, cudaReadModeElementType> tex_reorderBB;


__global__ void brownianBridgeGPU_reorder_texture(  
	int input_offset,                                                                   
	float* output_data,
	int rng_dim,
	int number_paths,
	int factors
	)
{

	int bx = blockIdx.x;
	int tx =  threadIdx.x;


	int gx = gridDim.x;
	int bwidth = blockDim.x;
	int width = gx*bwidth;



	int pathsPerThread = 1 + (( number_paths -1)/width);

	for (int l=0; l < pathsPerThread; ++l)
	{
		int pathNumber = width*l + bwidth*bx+tx;
		if (pathNumber < number_paths)
		{
			int input_data_offset =  input_offset + pathNumber;
			float* output_data_offset =  output_data + pathNumber*factors; // results in non-coalesced memory access 

			for (int i=0; i < rng_dim; ++i)
			{
				int outDim = dev_const_reorderArrayDim[i];
				int outFactor =dev_const_reorderArrayFactor[i];
				output_data_offset[outDim*number_paths*factors+outFactor] = tex1Dfetch(tex_reorderBB,input_data_offset+i*number_paths);

			}
		}
	}
}




__global__ void brownianBridgeGPU_reorder(                                                                    
	float* input_data,
	float* output_data,
	int rng_dim,
	int number_paths,
	int factors
	)
{

	int bx = blockIdx.x;
	int tx =  threadIdx.x;


	int gx = gridDim.x;
	int bwidth = blockDim.x;
	int width = gx*bwidth;



	int pathsPerThread = 1 + (( number_paths -1)/width);

	for (int l=0; l < pathsPerThread; ++l)
	{
		int pathNumber = width*l + bwidth*bx+tx;
		if (pathNumber < number_paths)
		{
			float* input_data_offset =  input_data + pathNumber;
			float* output_data_offset =  output_data + pathNumber*factors; // results in non-coalesced memory access 

			for (int i=0; i < rng_dim; ++i)
			{
				int outDim = dev_const_reorderArrayDim[i];
				int outFactor =dev_const_reorderArrayFactor[i];
				output_data_offset[outDim*number_paths*factors+outFactor] = input_data_offset[i*number_paths];

			}
		}
	}
}





extern "C"
void brownianBridgeMultiGPUReorder(float* input,
								   float* output,
								   int n_poweroftwo,
								   int factors,
								   int number_paths,
								   BrownianBridgeMultiDim<float>::ordering allocator,
								   bool useTextures
								   )
{


	int dimension;

	{
		BrownianBridgeMultiDim<float> bb(n_poweroftwo,factors, allocator);
		dimension = bb.innerBridge().dimensions();



		COPYCONSTANTMEMORYINT(dev_const_reorderArrayDim, bb.reorderingDimension());
		COPYCONSTANTMEMORYINT(dev_const_reorderArrayFactor, bb.reorderingFactor());
	}

	const int threadsperblock = 64;
	const int maxBlocks = 65535;

	// Set up the execution configuration
	dim3 dimGrid;
	dim3 dimBlock;

	dimGrid.x = 1+(number_paths-1)/threadsperblock;

	if (dimGrid.x > maxBlocks) 
		dimGrid.x=maxBlocks;

	// Fix the number of threads
	dimBlock.x = threadsperblock;

	CUT_CHECK_ERR("multi brownianBridgeGPU reordering kernel execution failed before entering kernel\n");

     ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);

	if
		(useTextures)
	{
		int inputOffset =0;
		// allocate array and copy image data
		cudaTextureFloatBinder reorderBinder(tex_reorderBB, input);

		brownianBridgeGPU_reorder_texture<<<dimGrid , dimBlock >>>(   
			inputOffset,
			output,
			dimension*factors,
			number_paths,
			factors);


		CUT_CHECK_ERR("multi brownianBridgeGPU kernel execution failed\n");



	}
	else
	{



		brownianBridgeGPU_reorder<<<dimGrid , dimBlock >>>(   
			input,
			output,
			dimension*factors,
			number_paths,
			factors);


		CUT_CHECK_ERR("multi brownianBridgeGPU kernel execution failed\n");

	}                                      

}
