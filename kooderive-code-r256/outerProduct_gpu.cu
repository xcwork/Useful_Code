//
//
//                                             outerProduct_gpu.cu
//
// (c) Mark Joshi 2012, 2014
// This code is released under the GNU public licence version 3



/*
cf Thread fence reduction kernel from CUDA SDK

Compute Outer product and reduce. 

Input data is a vector on the device 

We are working with floats so we don't want a straight-forward sum to avoid fp error.

We two matrices A and B of size  m \times paths, and n \times paths
we want to compute to AB^{t} in a fast and floating point effective manner, it will be of size m \times n

*/
#include <outerProduct_gpu.h>
#include <cuBlas.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include "cudaMacros.h"
#include <cutil_inline.h>
#include <gold/Timers.h>
#include <gold/Errors.h>
#include <gold/math/Power.h>
#include <CUDAConditionalMacros.h>

template <unsigned int blockSize>
__device__ void
	reduceElementBlock(volatile float *shared_data, float elementSum,  unsigned int threadIdent)
{
	shared_data[threadIdent] = elementSum;
	__syncthreads();

	// we do the reduction in shared memory for one element
	if (blockSize >= 512) 
	{ 
		if (threadIdent < 256) 
		{ 
			shared_data[threadIdent] = elementSum = elementSum + shared_data[threadIdent + 256]; 
		} 
		__syncthreads(); 
	}


	if (blockSize >= 256) 
	{ 
		if (threadIdent < 128) 
		{ 
			shared_data[threadIdent] = elementSum = elementSum + shared_data[threadIdent + 128]; 
		} 
		__syncthreads(); 
	}

	if (blockSize >= 128) 
	{ 
		if (threadIdent <  64) 
		{ 
			shared_data[threadIdent] = elementSum = elementSum + shared_data[threadIdent +  64]; 
		} 
		__syncthreads(); 
	}

	// we don't need sync threads for this part since the threads move in lock step
	if (threadIdent < 32)
	{
		if (blockSize >=  64) 
			shared_data[threadIdent] = elementSum = elementSum + shared_data[threadIdent + 32];  
		if (blockSize >=  32) 
			shared_data[threadIdent] = elementSum = elementSum + shared_data[threadIdent + 16]; 
		if (blockSize >=  16) 
			shared_data[threadIdent] = elementSum = elementSum + shared_data[threadIdent +  8];  
		if (blockSize >=   8) 
			shared_data[threadIdent] = elementSum = elementSum + shared_data[threadIdent +  4]; 
		if (blockSize >=   4) 
			shared_data[threadIdent] = elementSum = elementSum + shared_data[threadIdent +  2]; 
		if (blockSize >=   2) 
			shared_data[threadIdent] = elementSum = elementSum + shared_data[threadIdent +  1]; 
	}
}

// only does one element
template <unsigned int blockSize>
__device__ void
	reduceElementBlocks(const float* __restrict__ input_data_global1,
	const float* __restrict__ input_data_global2,
	float* __restrict__ output_data_global, int paths, int row, int column, int outloc)
{
	extern __shared__ float shared_data[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int threadIdent = threadIdx.x;

	unsigned int globalThreadIdent = blockIdx.x*blockSize + threadIdent;

	unsigned int path = globalThreadIdent;


	unsigned int gridSize = blockSize*gridDim.x;

	float elementSum = 0;

	// we reduce multiple elements per thread.  The number is determined by the 
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (path < paths)
	{    
		unsigned int dataPoint1 = path + paths*row;
		unsigned int dataPoint2 = path + paths*column;

		elementSum += LDG(input_data_global1+dataPoint1)*LDG(input_data_global2+dataPoint2);

		path += gridSize;
	} 

	// do reduction in shared mem
	reduceElementBlock<blockSize>(shared_data, elementSum, threadIdent);

	// write result for this element for this block to global mem 
	if (threadIdent == 0) 
		output_data_global[blockIdx.x+outloc] = shared_data[0];
}

__device__ unsigned int completionCount = 0;

template <unsigned int blockSize>
__global__ void reduceOuterProduct(const float* __restrict__ input_data_global1, 
								   const float* __restrict__ input_data_global2, 
								   float* __restrict__ output_data_workspace_global,
								   float* __restrict__ answer_global,
								   unsigned int paths, 
								   int size1, // ie m
								   int size2, // ie n
								   bool symmetric,
								   int row,
								   int col
								   )
{


	__syncthreads();

	reduceElementBlocks<blockSize>(input_data_global1,  input_data_global2, output_data_workspace_global, paths, row, col,0);

	const unsigned int threadIdent = threadIdx.x;
	__shared__ bool last;
	extern float __shared__ shared_mem[];


	//
	// The last block to finish will process all partial sums of this element
	//

	__threadfence();

	if( threadIdent==0 )
	{
		unsigned int numberDone = atomicInc(&completionCount, gridDim.x);

		last = (numberDone == gridDim.x-1); // atomicInc returns the value before the increment

	}
	__syncthreads();

	// The last block does all the work
	if( last )
	{
		int loc = threadIdent;
		float elementSum = 0;

		while (loc < gridDim.x)
		{         
			elementSum += output_data_workspace_global[loc];
			loc += blockSize;
		} 

		reduceElementBlock<blockSize>(shared_mem, elementSum, threadIdent);

		if( threadIdent==0 )  
		{
			answer_global[row*size2+col] = shared_mem[0];

			if (symmetric)
				answer_global[col*size1+row] = shared_mem[0];

			// return to initial conditions
			completionCount = 0; 

		}
	}
	__syncthreads();

}

//this places partially reduced sums in the partially_reduced_data array
// the number of outputs is blocks*rows*rows but only upper triangular part actually used
template<int blockSize>
__global__ void reducedOuterProductSymmetricAllFirstPass_kernel( float* inputData, 
																int rows,
																int paths, 
																float* partially_reduced_data,
																int width)
{
	extern float __shared__ floatData[];
	int threadId = threadIdx.x;
	int blockId = blockIdx.x;
	int globalThreadId = threadId+blockId*blockSize;

	for (int r=0; r < rows; ++r)
		for (int c=r; c<rows; ++c)
		{
			const float* input_loc_1 = inputData+r*paths;
			const float* input_loc_2 = inputData+c*paths;

			float x=0.0f;

			for (int p=globalThreadId; p < paths; p+=width)
				x+= LDG(input_loc_1+p)*LDG(input_loc_2+p);
				//	x+= (input_loc_1[p])*(input_loc_2[p]);

			// x now contains \sum x_i y_i over globalThreadId + i*width with i varying

			floatData[threadId] = x;

			__syncthreads();

			// the shared memory now contains the partial sums for the block

			if (blockSize >=1024)
			{
				if (threadIdx.x <512)
				{
					x += floatData[threadId+512];
					floatData[threadId] = x;

				}
				__syncthreads();

			}

			if (blockSize >=512)
			{
				if (threadIdx.x <256)
				{
					x += floatData[threadId+256];
					floatData[threadId] = x;
				}
				__syncthreads();


			}

			if (blockSize >=256)
			{
				if (threadIdx.x <128)
				{
					x += floatData[threadId+128];
					floatData[threadId] = x;
				}
				__syncthreads();
			}

			if (blockSize >=128)
			{
				if (threadIdx.x <64)
				{
					x += floatData[threadId+64];
					floatData[threadId] = x;
				}
				__syncthreads();
			}

			if (blockSize >=64)
			{
				if (threadIdx.x <32)
				{
					x += floatData[threadId+32];
					floatData[threadId] = x;
				}
				__syncthreads();
			}

			// note I am assuming you use a block size that is a power of 2 and is bigger than 16
			// arguably __ syncthreads is redundant since in same warp but this is not guaranteed for future architectures

			if (blockSize >=32)
			{
				if (threadIdx.x <16)
				{
					x += floatData[threadId+16];
					floatData[threadId] = x;
				}
				__syncthreads();
			}

			if (threadIdx.x <8)
			{
				x += floatData[threadId+8];
				floatData[threadId] = x;
			}
			__syncthreads();


			if (threadIdx.x <4)
			{
				x += floatData[threadId+4];
				floatData[threadId] = x;				

			}
			__syncthreads();

			if (threadIdx.x <2)
			{
				x += floatData[threadId+2];
				floatData[threadId] = x;				

			}

			__syncthreads();

			if (threadIdx.x ==0)
			{
				x += floatData[1];
				partially_reduced_data[blockId+gridDim.x*(r*rows+c)] = x;				

			}
		}
}

// each block does one matrix element
template<int blockSize>
__global__ void reduceAllBlocks(int rows, 
								float* out_matrix_loc,
								const float* block_reduced_data
								)
{
	extern float __shared__ floatData[];

	int threadId = threadIdx.x;
	int r = blockIdx.x;
	int c = blockIdx.y;

	if (r<=c)
	{
		// get data from global memory
		float x = LDG(block_reduced_data+(r*rows+c)*blockSize+threadId);

		reduceElementBlock<blockSize>(floatData, x,  threadId);

		if (threadId==0)
		{
			float res = *floatData;
			out_matrix_loc[r*rows+c]=res;
			out_matrix_loc[c*rows+r]=res;

		}
	}
}



// workspace_data_global needs to be at least as big as blocks*row*row
// it is only used for temporary data storage 

void reduceOuterProductSymmetric_gpu(int paths,
									 int row_size,
									 int threads, 
									 int blocks, 
									 float* input_data_global1,
									 float* answer_global,
									 float* workspace_data_global)
{
	// ensure that the config is one for which the code works
    
    int device;
    cudaDeviceProp deviceproperty;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceproperty, device);          

    int deviceThreads = static_cast<int>(deviceproperty.maxThreadsPerBlock);
    int deviceSharedMem = static_cast<int>(deviceproperty.sharedMemPerBlock);

	while (!IsPowerOfTwo(threads) ||threads >deviceThreads || threads * sizeof(float) >deviceSharedMem )
	{
		--threads;
	}

	if (threads < 32)
		threads = 32;

	if (blocks != threads)
		blocks = threads;


	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	int shared_memory_Size = threads * sizeof(float);
	int width=threads*blocks;

	//	reducedOuterProductSymmetricAllFirstPass<<< dimGrid, dimBlock, shared_memory_Size >>>(input_data_global1,row_size,paths, workspace_data_global, width); break;

	switch (threads)
	{
	case 1024:
		reducedOuterProductSymmetricAllFirstPass_kernel<1024> <<< dimGrid, dimBlock, shared_memory_Size >>>(input_data_global1,row_size,paths, workspace_data_global, width); break;
	case 512:
		reducedOuterProductSymmetricAllFirstPass_kernel<512> <<< dimGrid, dimBlock, shared_memory_Size >>>(input_data_global1,row_size,paths, workspace_data_global, width); break;
	case 256:
		reducedOuterProductSymmetricAllFirstPass_kernel<256> <<< dimGrid, dimBlock, shared_memory_Size >>>(input_data_global1,row_size,paths, workspace_data_global, width); break;
	case 128:
		reducedOuterProductSymmetricAllFirstPass_kernel<128> <<< dimGrid, dimBlock, shared_memory_Size >>>(input_data_global1,row_size,paths, workspace_data_global, width); break;
	case 64:
		reducedOuterProductSymmetricAllFirstPass_kernel<64> <<< dimGrid, dimBlock, shared_memory_Size >>>(input_data_global1,row_size,paths, workspace_data_global, width); break;
	case 32:
		reducedOuterProductSymmetricAllFirstPass_kernel<32><<< dimGrid, dimBlock, shared_memory_Size >>>(input_data_global1,row_size,paths, workspace_data_global, width); break;

	default:
		GenerateError("block size not supported");
	}
	// the first pass has put the contents of the reduced products into data of size blocks, we now need to reduce those blocks
	cutilSafeCall(cudaThreadSynchronize()); 

	dim3 dimGrid2(row_size, row_size, 1);
	threads=blocks;
	int shared_memory_Size2 = threads * sizeof(float);

	dim3 dimBlocks(threads,1,1);
	switch (threads)
	{
	case 1024:
		reduceAllBlocks<1024><<< dimGrid2, dimBlock, shared_memory_Size2 >>>(row_size,answer_global,workspace_data_global); break;
	case 512:
		reduceAllBlocks<512><<< dimGrid2, dimBlock, shared_memory_Size2 >>>(row_size,answer_global,workspace_data_global); break;
	case 256:
		reduceAllBlocks<256><<< dimGrid2, dimBlock, shared_memory_Size2 >>>(row_size,answer_global,workspace_data_global); break;
	case 128:
		reduceAllBlocks<128><<< dimGrid2, dimBlock, shared_memory_Size2 >>>(row_size,answer_global,workspace_data_global); break;
	case 64:
		reduceAllBlocks<64><<< dimGrid2, dimBlock, shared_memory_Size2 >>>(row_size,answer_global,workspace_data_global); break;
	case 32:
		reduceAllBlocks<32><<< dimGrid2, dimBlock, shared_memory_Size2 >>>(row_size,answer_global,workspace_data_global); break;

	}
	cutilSafeCall(cudaThreadSynchronize()); 
}


//this places partially reduced sums in the partially_reduced_data array
// the number of outputs is blocks*rows*columns
template<int blockSize>
__global__ void reducedOuterProductAllFirstPass_kernel( float* inputData1,
													   float* inputData2,
													   int rows,
													   int columns,
													   int paths, 
													   float* partially_reduced_data,
													   int width)
{
	extern float __shared__ floatData[];
	int threadId = threadIdx.x;
	int blockId = blockIdx.x;
	int globalThreadId = threadId+blockId*blockSize;

	for (int r=0; r < rows; ++r)
		for (int c=0; c<columns; ++c)
		{
			const float* input_loc_1 = inputData1+r*paths;
			const float* input_loc_2 = inputData2+c*paths;

			float x=0.0f;

			for (int p=globalThreadId; p < paths; p+=width)
			    x+= LDG(input_loc_1+p)*LDG(input_loc_2+p);
			//		x+= input_loc_1[p]*input_loc_2[p];

			// x now contains \sum x_i y_i over globalThreadId + i*width with i varying

			floatData[threadId] = x;

			__syncthreads();

			// the shared memory now contains the partial sums for the block

			if (blockSize >=1024)
			{
				if (threadIdx.x <512)
				{
					x += floatData[threadId+512];
					floatData[threadId] = x;

				}
				__syncthreads();

			}

			if (blockSize >=512)
			{
				if (threadIdx.x <256)
				{
					x += floatData[threadId+256];
					floatData[threadId] = x;
				}
				__syncthreads();


			}

			if (blockSize >=256)
			{
				if (threadIdx.x <128)
				{
					x += floatData[threadId+128];
					floatData[threadId] = x;
				}
				__syncthreads();
			}

			if (blockSize >=128)
			{
				if (threadIdx.x <64)
				{
					x += floatData[threadId+64];
					floatData[threadId] = x;
				}
				__syncthreads();
			}

			if (blockSize >=64)
			{
				if (threadIdx.x <32)
				{
					x += floatData[threadId+32];
					floatData[threadId] = x;
				}
				__syncthreads();
			}

			// note I am assuming you use a block size that is a power of 2 and is bigger than 16
			// arguably __ syncthreads is redundant since in same warp but this is not guaranteed for future architectures

			if (blockSize >=32)
			{
				if (threadIdx.x <16)
				{
					x += floatData[threadId+16];
					floatData[threadId] = x;
				}
				__syncthreads();
			}

			if (threadIdx.x <8)
			{
				x += floatData[threadId+8];
				floatData[threadId] = x;
			}
			__syncthreads();


			if (threadIdx.x <4)
			{
				x += floatData[threadId+4];
				floatData[threadId] = x;				

			}
			__syncthreads();

			if (threadIdx.x <2)
			{
				x += floatData[threadId+2];
				floatData[threadId] = x;				

			}

			__syncthreads();

			if (threadIdx.x ==0)
			{
				x += floatData[1];
				partially_reduced_data[blockId+gridDim.x*(r*columns+c)] = x;				

			}
		}
}


// each block does one matrix element
template<int blockSize>
__global__ void reduceAllBlocksAsymmetric(int rows,
										  int columns,
										  float* out_matrix_loc,
										  const float* block_reduced_data
										  )
{
	extern float __shared__ floatData[];

	int threadId = threadIdx.x;
	int r = blockIdx.x;
	int c = blockIdx.y;

	if (r<=rows && c<= columns)
	{
		// get data from global memory
		float x = LDG(block_reduced_data+(r*columns+c)*blockSize+threadId);
		reduceElementBlock<blockSize>(floatData, x,  threadId);

		if (threadId==0)
		{
			float res = *floatData;
			out_matrix_loc[r*columns+c]=res;			
		}
	}
}
void reduceOuterProduct_gpu(int paths,
							int row_size,
							int col_size,
							bool symmetric,
							int threads, 
							int blocks, 
							float* input_data_global1,
							float* input_data_global2,
							float* answer_global,
							float* workspace_data_global)
{
	if (symmetric)
	{
		reduceOuterProductSymmetric_gpu(paths,
			row_size,
			threads, 
			blocks, 
			input_data_global1,
			answer_global,
			workspace_data_global);
	}
	else
	{
		reducedOuterProduct_gpu( paths,
			row_size,
			col_size,
			threads, 
			blocks, 
			input_data_global1,
			input_data_global2,
			answer_global,
			workspace_data_global);
	}
}

// workspace_data_global needs to be at least as big as blocks*row*row
// it is only used for temporary data storage 

void reducedOuterProduct_gpu(int paths,
							 int row_size,
							 int col_size,
							 int threads, 
							 int blocks, 
							 float* input_data_global1,
							 float* input_data_global2,
							 float* answer_global,
							 float* workspace_data_global)
{
	//std::cout << " calling new outer product...";

	// ensure that the config is one for which the code works

	while (!IsPowerOfTwo(threads))
	{
		--threads;
	}

	if (threads < 32)
		threads = 32;

	if (blocks != threads)
		blocks = threads;

	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	int shared_memory_Size = threads * sizeof(float);
	int width=threads*blocks;

	//	reducedOuterProductSymmetricAllFirstPass<<< dimGrid, dimBlock, shared_memory_Size >>>(input_data_global1,row_size,paths, workspace_data_global, width); break;

	switch (threads)
	{
	case 1024:
		reducedOuterProductAllFirstPass_kernel<1024> <<< dimGrid, dimBlock, shared_memory_Size >>>(input_data_global1,input_data_global2,row_size,col_size,paths, workspace_data_global, width); break;
	case 512:
		reducedOuterProductAllFirstPass_kernel<512> <<< dimGrid, dimBlock, shared_memory_Size >>>(input_data_global1,input_data_global2,row_size,col_size,paths, workspace_data_global, width); break;
	case 256:
		reducedOuterProductAllFirstPass_kernel<256> <<< dimGrid, dimBlock, shared_memory_Size >>>(input_data_global1,input_data_global2,row_size,col_size,paths, workspace_data_global, width); break;
	case 128:
		reducedOuterProductAllFirstPass_kernel<128> <<< dimGrid, dimBlock, shared_memory_Size >>>(input_data_global1,input_data_global2,row_size,col_size,paths, workspace_data_global, width); break;
	case 64:
		reducedOuterProductAllFirstPass_kernel<64> <<< dimGrid, dimBlock, shared_memory_Size >>>(input_data_global1,input_data_global2,row_size,col_size,paths, workspace_data_global, width); break;
	case 32:
		reducedOuterProductAllFirstPass_kernel<32><<< dimGrid, dimBlock, shared_memory_Size >>>(input_data_global1,input_data_global2,row_size,col_size,paths, workspace_data_global, width); break;

	default:
		GenerateError("block size not supported");
	}

	cutilSafeCall(cudaThreadSynchronize()); 

	// the first pass has put the contents of the reduced products into data of size blocks, we now need to reduce those blocks

	dim3 dimGrid2(row_size, col_size, 1);
	threads=blocks;
	dim3 dimBlocks(threads,1,1);
	int shared_memory_Size2 = threads * sizeof(float);
	switch (threads)
	{
	case 1024:
		reduceAllBlocksAsymmetric<1024><<< dimGrid2, dimBlock, shared_memory_Size2 >>>(row_size, col_size,answer_global,workspace_data_global); break;
	case 512:
		reduceAllBlocksAsymmetric<512><<< dimGrid2, dimBlock, shared_memory_Size2 >>>(row_size, col_size,answer_global,workspace_data_global); break;
	case 256:
		reduceAllBlocksAsymmetric<256><<< dimGrid2, dimBlock, shared_memory_Size2 >>>(row_size, col_size,answer_global,workspace_data_global); break;
	case 128:
		reduceAllBlocksAsymmetric<128><<< dimGrid2, dimBlock, shared_memory_Size2 >>>(row_size, col_size,answer_global,workspace_data_global); break;
	case 64:
		reduceAllBlocksAsymmetric<64><<< dimGrid2, dimBlock, shared_memory_Size2 >>>(row_size, col_size,answer_global,workspace_data_global); break;
	case 32:
		reduceAllBlocksAsymmetric<32><<< dimGrid2, dimBlock, shared_memory_Size2 >>>(row_size, col_size,answer_global,workspace_data_global); break;

	}
	cutilSafeCall(cudaThreadSynchronize()); 
}


void reduceOuterProductSymmetric_cublas(int paths,
										int row_size,
										float alpha,
										float beta,
										float* input_data_global,
										float* answer_global)
{
	cublasInit();

	reduceOuterProductSymmetric_cublas_initted(paths,
		row_size,
		alpha,
		beta,
		input_data_global,
		answer_global);


	cublasShutdown();

}


void reduceOuterProduct_cublas(int paths,
							   int row_size,
							   int col_size,
							   float alpha,
							   float beta,
							   float* input_data_global1,
							   float* input_data_global2,
							   float* answer_global)
{
	cublasInit();

	reduceOuterProduct_cublas_initted(paths,
		row_size,
		col_size,
		alpha,
		beta,
		input_data_global1,
		input_data_global2,
		answer_global);


	cublasShutdown();

}


void reduceOuterProductSymmetric_cublas_initted(int paths,
												int row_size,
												float alpha,
												float beta,
												float* input_data_global,
												float* answer_global)
{


	char transpose1='t';;
	char transpose2='n';


	int m = row_size;
	int n= row_size;
	int k = paths;


	int lda = k;
	int ldb = k;
	int ldc = row_size;

	cublasSgemm( transpose1, transpose2, m,n ,k , alpha, input_data_global, lda, input_data_global, ldb, beta, answer_global,ldc);

	CUT_CHECK_ERR("cublasSgemm execution failed\n");




}



void reduceOuterProduct_cublas_initted(int paths,
									   int row_size,
									   int col_size,
									   float alpha,
									   float beta,
									   float* input_data_global1,
									   float* input_data_global2,
									   float* answer_global)
{

	char transpose1='t';;
	char transpose2='n';


	int m = row_size;
	int n= col_size;
	int k = paths;


	int lda = k;
	int ldb = k;
	int ldc = row_size;

	cublasSgemm( transpose1, transpose2, m,n ,k , alpha, input_data_global1, 
		lda, input_data_global2, ldb, beta, answer_global,ldc);

	CUT_CHECK_ERR("cublasSgemm execution failed\n");




}

/*

void reduceOuterProductSymmetric_cublas_initted(int paths,
int row_size,
double alpha,
double beta,
double* input_data_global,
double* answer_global)
{


char transpose1='t';;
char transpose2='n';


int m = row_size;
int n= row_size;
int k = paths;


int lda = k;
int ldb = k;
int ldc = row_size;

cublasDgemm( transpose1, transpose2, m,n ,k , alpha, input_data_global, lda, input_data_global, ldb, beta, answer_global,ldc);

CUT_CHECK_ERR("cublasDgemm execution failed\n");




}
void reduceOuterProduct_cublas_initted(int paths,
int row_size,
int col_size,
double alpha,
double beta,
double* input_data_global1,
double* input_data_global2,
double* answer_global)
{

char transpose1='t';;
char transpose2='n';


int m = row_size;
int n= col_size;
int k = paths;


int lda = k;
int ldb = k;
int ldc = row_size;

cublasDgemm( transpose1, transpose2, m,n ,k , alpha, input_data_global1, 
lda, input_data_global2, ldb, beta, answer_global,ldc);

CUT_CHECK_ERR("cublasDgemm execution failed\n");




}
*/


__global__ void PointwiseProduct_kernel(
	const float* data1,
	const float* data2,
	float* data3,
	int paths,
	int pathsPerThread,
	int width)
{
	int bx = blockIdx.x;

	int bwidth = blockDim.x;
	int tx = threadIdx.x;

	for (int l=0; l < pathsPerThread; ++l)
	{
		int pathNumber = width*l + bwidth*bx+tx;
		if (pathNumber < paths)
			data3[pathNumber] = LDG(data1+pathNumber)*LDG(data2+pathNumber);
	}
}

double PointwiseProduct_gpu(int blocks,
							int threads,
							int paths,
							const float* in1_global,
							const float* in2_global,
							float* out_global)
{
	int width = blocks*threads;
	int pathsPerThread = 1 + (( paths -1)/width);
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	Timer h1;

	PointwiseProduct_kernel<<<dimBlock,dimGrid >>>(
		in1_global,
		in2_global,
		out_global,
		paths,
		pathsPerThread,
		width);

	cudaThreadSynchronize();
	double t=h1.timePassed();
	return t;
}


__global__ void PointwiseSymmetricUpperTriangular_kernel(
	const float* data,
	float* data_out,
	int paths,
	int pathsPerThread,
	int width,
	int rows)
{
	int bx = blockIdx.x;

	int bwidth = blockDim.x;
	int tx = threadIdx.x;

	for (int r=0; r < rows; ++r)
	{
		float* data_out_loc = data_out+paths*rows*r+paths*r;

		for (int l=0; l < pathsPerThread; ++l)
		{
			int pathNumber = width*l + bwidth*bx+tx;
			if (pathNumber < paths)
			{
				float x= //data[pathNumber+paths*r];
					LDG(data+pathNumber+paths*r);
				data_out_loc[pathNumber] = x*x;

				for (int c=r+1; c < rows; ++c)
				{
					data_out_loc +=paths;
					float y = //data[pathNumber+paths*c];
						LDG(data+pathNumber+paths*c);
					data_out_loc[pathNumber] = x*y;


				}
			}
		}

	}

}



double PointwiseProductSymmetricUpperTriangular_gpu(int blocks,
													int threads,
													int paths,
													int rows,
													const float* in_global,
													float* out_global)
{
	int width = blocks*threads;
	int pathsPerThread = 1 + (( paths -1)/width);
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	Timer h1;

	PointwiseSymmetricUpperTriangular_kernel<<<dimBlock,dimGrid >>>(
		in_global,
		out_global,
		paths,
		pathsPerThread,
		width,
		rows);

	cudaThreadSynchronize();
	double t=h1.timePassed();
	return t;
}

__global__ void PointwiseProductMultipleFirst_kernel(
	const float* data1,
	const float* data2,
	float* data3,
	int paths,
	int pathsPerThread,
	int width, 
	int rows)
{
	int bx = blockIdx.x;

	int bwidth = blockDim.x;
	int tx = threadIdx.x;

	for (int l=0; l < pathsPerThread; ++l)
	{
		int pathNumber = width*l + bwidth*bx+tx;
		if (pathNumber < paths)
		{
			float y= LDG(data2+pathNumber);
			for (int i=0; i <rows; ++i)
				data3[pathNumber+i*paths] = LDG(data1+i*paths+pathNumber)*y;
		}
	}
}

double PointwiseProductMultipleFirst_gpu(int blocks,
										 int threads,
										 int paths,
										 const float* in1_global,
										 const float* in2_global,
										 float* out_global,
										 int rows)
{
	int width = blocks*threads;
	int pathsPerThread = 1 + (( paths -1)/width);
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	Timer h1;

	PointwiseProductMultipleFirst_kernel<<<dimBlock,dimGrid >>>(
		in1_global,
		in2_global,
		out_global,
		paths,
		pathsPerThread,
		width, 
		rows);

	cudaThreadSynchronize();
	double t=h1.timePassed();
	return t;
}

