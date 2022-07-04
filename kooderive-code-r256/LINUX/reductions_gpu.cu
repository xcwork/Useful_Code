#include <cuda_runtime.h>
#include <cutil.h>
#include "cudaMacros.h"
#include <cutil_inline.h>
#include <gold/Timers.h>
#include <iostream>
#include <reduction_thread_fence.h>
#include <CUDAConditionalMacros.h>
texture<float, 1, cudaReadModeElementType> tex_reduction;

__global__ void PartialReduce_8_kernel(
	const float* data,
	float* data_out,
	int points,
	int pointsPerThread,
	int width)
{
	int bx = blockIdx.x;

	int bwidth = blockDim.x;
	int tx = threadIdx.x;

	for (int l=0; l < pointsPerThread; ++l)
	{
		int pathNumber = width*l + bwidth*bx+tx;
		if (pathNumber*8 < points)
		{
			const float* dataP = data+pathNumber*8;
			float x0= LDG(dataP);

			x0+=LDG(dataP+1);

			float x2= LDG(dataP+2);
			x2+= LDG(dataP+3);

			float x4= LDG(dataP+4);
			x4+= LDG(dataP+5);

			float x6= LDG(dataP+6);
			x6+=LDG(dataP+7);

			x0+=x2;
			x4+=x6;
			x0+=x4;

			data_out[pathNumber]=x0;
		}
	}
}


__global__ void PartialReduce_8_single_kernel(
	const float* data,
	float* data_out)
{
	int bx = blockIdx.x;

	int bwidth = blockDim.x;
	int tx = threadIdx.x;

	int pathNumber =  bwidth*bx+tx;
	const float* dataP = data+pathNumber*8;
	float x0= LDG(dataP);

	x0+=LDG(dataP+1);

	float x2= LDG(dataP+2);
	x2+= LDG(dataP+3);

	float x4= LDG(dataP+4);
	x4+= LDG(dataP+5);

	float x6= LDG(dataP+6);
	x6+=LDG(dataP+7);

	x0+=x2;
	x4+=x6;
	x0+=x4;

	data_out[pathNumber]=x0;

}


__global__ void PartialReduce_8_single_texture_kernel(
	float* data_out)
{
	int bx = blockIdx.x;

	int bwidth = blockDim.x;
	int tx = threadIdx.x;

	int pathNumber =  bwidth*bx+tx;

	int off = pathNumber*8;
	float x0= tex1Dfetch(tex_reduction,off);

	++off;
	x0+=tex1Dfetch(tex_reduction,off);
	++off;
	float x2= tex1Dfetch(tex_reduction,off);
	++off;
	x2+= tex1Dfetch(tex_reduction,off);
	++off;
	float x4= tex1Dfetch(tex_reduction,off);
	++off;
	x4+= tex1Dfetch(tex_reduction,off);
	++off;
	float x6=tex1Dfetch(tex_reduction,off);
	++off;
	x6+=tex1Dfetch(tex_reduction,off);

	x0+=x2;
	x4+=x6;
	x0+=x4;

	data_out[pathNumber]=x0;

}
double PartialReduce8_gpu(const float* data,
						  float* data_out,
						  int points,
						  int blocks,
						  int threads)
{
	int width = blocks*threads;
	int pointsPerThread = 1 + (( points -1)/width);
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	Timer h1;

	bool textures = false;

//	std::cout << width << " " << 8*points << "\n";

	if (width*8 == points)
	{
		if (textures)
		{
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

			// set texture parameters
			tex_reduction.addressMode[0] = cudaAddressModeWrap;
			tex_reduction.addressMode[1] = cudaAddressModeWrap;
			tex_reduction.filterMode = cudaFilterModeLinear;
			tex_reduction.normalized = false;    // access with normalized texture coordinates

			// Bind the array to the texture
			cudaBindTexture( NULL, tex_reduction, data, channelDesc);


			PartialReduce_8_single_texture_kernel<<<dimBlock,dimGrid >>>(
				data_out);

			cudaThreadSynchronize();
			cudaUnbindTexture(tex_reduction);


		}
		else

			PartialReduce_8_single_kernel<<<dimBlock,dimGrid >>>(
			data,
			data_out);
	}
	else
		PartialReduce_8_kernel<<<dimBlock,dimGrid >>>(
		data,
		data_out,
		points,
		pointsPerThread,
		width);

	cudaThreadSynchronize();
	double t=h1.timePassed();
	return t;
}

const float* PartialReduce_using8s_gpu(const float* data,
									   float* workspace, // at least size points
									   int points,
									   int threads,
									   int stopPoint)
{
	int pointsLeft =points;

	const float* data_loc = data;
	float* new_data_loc = workspace;

	while (pointsLeft >stopPoint)
	{
		int newPoints = pointsLeft/8;
		int pointsDiscarded = pointsLeft % 8;

		if (pointsDiscarded>0)
			std::cout << " warning points discarded in partial reduce." << pointsLeft << " " << newPoints << " " << pointsDiscarded << "\n";

		while (threads > pointsLeft)
			threads/=2;

		int blocks = 1+(pointsLeft-1)/(threads*8);

		PartialReduce8_gpu(data_loc,new_data_loc,pointsLeft,blocks,threads);

		data_loc = new_data_loc;
		new_data_loc += newPoints;
		pointsLeft = newPoints;

		//	std::cout << pointsLeft << "\n";
	}

	return data_loc;

}
/////////////////////////////////////////////////////////////////////////////////////////////

__global__ void PartialReduce_16_kernel(
	const float* data,
	float* data_out,
	int points,
	int pointsPerThread,
	int width)
{
	int bx = blockIdx.x;

	int bwidth = blockDim.x;
	int tx = threadIdx.x;

	for (int l=0; l < pointsPerThread; ++l)
	{
		int pathNumber = width*l + bwidth*bx+tx;
		if (pathNumber*16 < points)
		{
			const float* dataP = data+pathNumber*16;
			float x0= LDG(dataP);
			{
				x0+=LDG(dataP+1);

				float x2= LDG(dataP+2);
				x2+= LDG(dataP+3);

				float x4= LDG(dataP+4);
				x4+= LDG(dataP+5);

				float x6= LDG(dataP+6);
				x6+=LDG(dataP+7);

				x0+=x2;
				x4+=x6;
				x0+=x4;
			}
			dataP+=8;
			float y0=LDG(dataP);
			{
				y0+=LDG(dataP+1);

				float y2= LDG(dataP+2);
				y2+= LDG(dataP+3);

				float y4= LDG(dataP+4);
				y4+= LDG(dataP+5);

				float y6= LDG(dataP+6);
				y6+=LDG(dataP+7);

				y0+=y2;
				y4+=y6;
				y0+=y4;
			}
			x0+=y0;
			data_out[pathNumber]=x0;
		}
	}
}

double PartialReduce16_gpu(const float* data,
						   float* data_out,
						   int points,
						   int blocks,
						   int threads)
{
	int width = blocks*threads;
	int pointsPerThread = 1 + (( points -1)/width);
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	Timer h1;

	PartialReduce_16_kernel<<<dimBlock,dimGrid >>>(
		data,
		data_out,
		points,
		pointsPerThread,
		width);

	cudaThreadSynchronize();
	double t=h1.timePassed();
	return t;
}

const float* PartialReduce_using16s_gpu(const float* data,
										float* workspace, // at least size points
										int points,
										int threads)
{
	int pointsLeft =points;

	const float* data_loc = data;
	float* new_data_loc = workspace;

	while (pointsLeft >1)
	{
		int newPoints = pointsLeft/16;
		int pointsDiscarded = pointsLeft % 16;

		if (pointsDiscarded>0)
			std::cout << " warning points discarded in partial reduce.";

		while (threads > pointsLeft)
			threads/=2;

		int blocks = 1+(pointsLeft-1)/(threads*16);

		PartialReduce16_gpu(data_loc,new_data_loc,pointsLeft,blocks,threads);

		data_loc = new_data_loc;
		new_data_loc += newPoints;
		pointsLeft = newPoints;

		//	std::cout << pointsLeft << "\n";
	}

	return data_loc;

}
///////////////////////

