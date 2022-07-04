//
//
//                                                                                                     cashFlowDiscounting_gpu
//
//
//

// (c) Mark Joshi 2010,2011,2013
// This code is released under the GNU public licence version 3

#include <cashFlowDiscounting_gpu.h>


#include <cudaWrappers/cudaTextureBinder.h>
#include <gpuCompatibilityCheck.h>
#include <gold/Errors.h>
#include "cudaMacros.h"

texture<int, 1> tex_genIndex;
texture<int, 1> tex_firstIndex;
texture<int, 1> tex_secondIndex;
texture<int, 1> tex_targetIndex;
texture<float, 1, cudaReadModeElementType> tex_theta;
texture<float, 1, cudaReadModeElementType> tex_discounts;




__global__ void cashFlowDiscounting_kernel(const float* __restrict__ genFlows, 
										   const float* __restrict__ numeraireValues,
										   int paths, int numberSteps,
										   int numberDfsPerStep,
										   float* discountedFlows,  // output
										   float* summedDiscountedFlows // output
										   )
{
	int bx = blockIdx.x;
	int tx =  threadIdx.x;

	int gx = gridDim.x;
	int bwidth = blockDim.x;
	int width = gx*bwidth;

	int pathsPerThread = 1 + (( paths -1)/width);


	for (int l=0; l < pathsPerThread; ++l)
	{
		int pathNumber = width*l + bwidth*bx+tx;
		int dataOffset = pathNumber;
		int dataOffset2 = pathNumber;

		if (pathNumber < paths)
		{
			float total=0.0;

			for (int i =0; i < numberSteps; ++i)
			{
				int dfIndex1 = tex1Dfetch(tex_firstIndex,i);
				int dfIndex2 = tex1Dfetch(tex_secondIndex,i);
				float theta =   tex1Dfetch(tex_theta,i);
				float numeraireValue = numeraireValues[dataOffset];

				float rdfToNow = tex1Dfetch(tex_discounts,dataOffset2+i*paths);
				float rdfTo1 = tex1Dfetch(tex_discounts,dataOffset2+dfIndex1*paths);
				float rdfTo2 = tex1Dfetch(tex_discounts,dataOffset2+dfIndex2*paths);
				rdfTo1 /= rdfToNow;
				rdfTo2 /= rdfToNow;

				float rdf = pow(rdfTo1, 1.0f-theta)*pow(rdfTo2, theta);

				float dfTonumeraire = rdf/numeraireValue;

				float flow = genFlows[dataOffset]*dfTonumeraire;
				total+=flow;

				discountedFlows[dataOffset] =flow; 

				dataOffset += paths;
				dataOffset2 += paths*numberDfsPerStep;
			}
			summedDiscountedFlows[pathNumber] =  total;        
		}
	}
}




__global__ void cashFlowDiscounting_shared_kernel(float* genFlows, 
												  float* numeraireValues,
												  int paths, 
												  int numberSteps,
												  int numberDfsPerStep,
												  int* firstIndex_global,
												  int* secondIndex_global,
												  float* theta_global,
												  float* discountedFlows,  // output
												  float* summedDiscountedFlows // output
												  )
{
	int bx = blockIdx.x;
	int tx =  threadIdx.x;

	int gx = gridDim.x;
	int bwidth = blockDim.x;
	int width = gx*bwidth;

	int pathsPerThread = 1 + (( paths -1)/width);

	extern __shared__ float data_shared[];
	float* theta_s = data_shared;
	int* firstIndex_s = reinterpret_cast<int*>(theta_s+numberSteps); 
	int* secondIndex_s = reinterpret_cast<int*>(firstIndex_s+numberSteps); 


	while (tx < numberSteps)
	{
		theta_s[tx] = theta_global[tx];
		firstIndex_s[tx] = firstIndex_global[tx];
		secondIndex_s[tx] = secondIndex_global[tx];
		tx += blockDim.x;
	}

	tx = threadIdx.x;

    __syncthreads();

	for (int l=0; l < pathsPerThread; ++l)
	{
		int pathNumber = width*l + bwidth*bx+tx;
		int dataOffset = pathNumber;
		int dataOffset2 = pathNumber;

		if (pathNumber < paths)
		{
			float total=0.0;

			for (int i =0; i < numberSteps; ++i)
			{
				int dfIndex1 = firstIndex_s[i];
				int dfIndex2 = secondIndex_s[i];
				float theta =   theta_s[i];
				float numeraireValue = numeraireValues[dataOffset];

				float rdfToNow = tex1Dfetch(tex_discounts,dataOffset2+i*paths);
				float rdfTo1 = tex1Dfetch(tex_discounts,dataOffset2+dfIndex1*paths);
				float rdfTo2 = tex1Dfetch(tex_discounts,dataOffset2+dfIndex2*paths);
				rdfTo1 /= rdfToNow;
				rdfTo2 /= rdfToNow;

				float rdf = pow(rdfTo1, 1.0f-theta)*pow(rdfTo2, theta);

				float dfTonumeraire = rdf/numeraireValue;

				float flow = genFlows[dataOffset]*dfTonumeraire;
				total+=flow;

				discountedFlows[dataOffset] =flow; 

				dataOffset += paths;
				dataOffset2 += paths*numberDfsPerStep;
			}
			summedDiscountedFlows[pathNumber] =  total;        
		}
	}
}


__global__ void cashFlowDiscounting_zerotextures_kernel(float* genFlows, 
														float* numeraireValues,
														float* discounts_global,
														int paths, 
														int numberSteps,
														int numberDfsPerStep,
														int* firstIndex_global,
														int* secondIndex_global,
														float* theta_global,
														float* discountedFlows,  // output
														float* summedDiscountedFlows // output
														)
{
	int bx = blockIdx.x;
	int tx =  threadIdx.x;

	int gx = gridDim.x;
	int bwidth = blockDim.x;
	int width = gx*bwidth;

	int pathsPerThread = 1 + (( paths -1)/width);

	extern __shared__ float data_shared[];
	float* theta_s = data_shared;
	int* firstIndex_s = reinterpret_cast<int*>(theta_s+numberSteps); 
	int* secondIndex_s = reinterpret_cast<int*>(firstIndex_s+numberSteps); 


	while (tx < numberSteps)
	{
		theta_s[tx] = theta_global[tx];
		firstIndex_s[tx] = firstIndex_global[tx];
		secondIndex_s[tx] = secondIndex_global[tx];
		tx += blockDim.x;
	}

	tx = threadIdx.x;
    __syncthreads();


	for (int l=0; l < pathsPerThread; ++l)
	{
		int pathNumber = width*l + bwidth*bx+tx;
		int dataOffset = pathNumber;
		int dataOffset2 = pathNumber;

		if (pathNumber < paths)
		{
			float total=0.0;

			for (int i =0; i < numberSteps; ++i)
			{
				int dfIndex1 = firstIndex_s[i];
				int dfIndex2 = secondIndex_s[i];
				float theta =   theta_s[i];
				float numeraireValue = numeraireValues[dataOffset];

				float rdfToNow =discounts_global[dataOffset2+i*paths];
				 //tex1Dfetch(tex_discounts,dataOffset2+i*paths);
				float rdfTo1 =discounts_global[dataOffset2+dfIndex1*paths];
				   //tex1Dfetch(tex_discounts,dataOffset2+dfIndex1*paths);
				float rdfTo2 = discounts_global[dataOffset2+dfIndex2*paths];
				  // tex1Dfetch(tex_discounts,dataOffset2+dfIndex2*paths);

				rdfTo1 /= rdfToNow;
				rdfTo2 /= rdfToNow;

				float rdf = pow(rdfTo1, 1.0f-theta)*pow(rdfTo2, theta);

				float dfTonumeraire = rdf/numeraireValue;

				float flow = genFlows[dataOffset]*dfTonumeraire;
				total+=flow;

				discountedFlows[dataOffset] =flow; 

				dataOffset += paths;
				dataOffset2 += paths*numberDfsPerStep;
			}
			summedDiscountedFlows[pathNumber] =  total;        
		}
	}
}





extern"C"
void cashFlowDiscounting_gpu(
							 int* firstIndex, 
							 int* secondIndex,
							 float* theta, 
							 float* discountRatios, 
							 float* genFlows, 
							 float* numeraireValues,
							 int paths, 
							 int numberSteps, 
							 bool useShared,
                             bool useTextures,
							 float* discountedFlows, // output
							 float* summedDiscountedFlows) // output
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

	int numberDfs = numberSteps+1;
	cudaTextureFloatBinder texDiscountsBinder(tex_discounts, discountRatios);

	bool notextures=!useTextures;
    CUT_CHECK_ERR("cashFlowDiscounting execution failed before entering kernel\n");

	if (notextures)
	{
		int sharedSize = 2*sizeof(int)*numberSteps+sizeof(float)*numberSteps;
      
         float excessMemRatio,excessSharedRatio;

          ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid,
                                  1,//   size_t estimatedMem, 
                                    sharedSize,  
                                    excessMemRatio, 
                                    excessSharedRatio);

          if (excessSharedRatio >0.0)
          {
                GenerateError("too much shared memory required for cashFlowDiscounting_zerotextures_kernel");
          }
          else
          {
    		cashFlowDiscounting_zerotextures_kernel<<<dimGrid , dimBlock, sharedSize >>>( genFlows, 
			numeraireValues,
			discountRatios,
			paths, 
			numberSteps,
			numberDfs,
			firstIndex,
			secondIndex,
			theta,
			discountedFlows,  // output
			summedDiscountedFlows // output
			);
          }
	}
	else
	{
		
		if (useShared)
		{
			int sharedSize = 2*sizeof(int)*numberSteps+sizeof(float)*numberSteps;
			cashFlowDiscounting_shared_kernel<<<dimGrid , dimBlock, sharedSize >>>( genFlows, 
				numeraireValues,
				paths, 
				numberSteps,
				numberDfs,
				firstIndex,
				secondIndex,
				theta,
				discountedFlows,  // output
				summedDiscountedFlows // output
				);

		}
		else
		{
			cudaTextureIntBinder firstBinder(tex_firstIndex,firstIndex);
			cudaTextureIntBinder secondBinder(tex_secondIndex,secondIndex);
			cudaTextureFloatBinder texBinder(tex_theta, theta);

			cashFlowDiscounting_kernel<<<dimGrid , dimBlock >>>(genFlows, numeraireValues,paths,numberSteps, numberDfs, discountedFlows, summedDiscountedFlows);
		}
	}

	   CUT_CHECK_ERR("cashFlowDiscounting execution failed after entering kernel\n");

}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////


__global__ void cashFlowDiscounting_partial_kernel(int numberFlowsPerPath,
												   float* genFlows, 
												   float* numeraireValues,
												   int paths, 
												   int numberSteps,
												   int numberDfsPerStepPerPath,
												   float* discountedFlows // for output
												   )
{
	int bx = blockIdx.x;
	int tx =  threadIdx.x;

	int gx = gridDim.x;
	int bwidth = blockDim.x;
	int width = gx*bwidth;

	int pathsPerThread = 1 + (( paths -1)/width);


	for (int l=0; l < pathsPerThread; ++l)
	{
		int pathNumber = width*l + bwidth*bx+tx;
		int dataOffset = pathNumber;


		if (pathNumber < paths)
		{              
			for (int flowNumber =0; flowNumber < numberFlowsPerPath; ++flowNumber)
			{
				int dfIndex1 = tex1Dfetch(tex_firstIndex,flowNumber);
				int dfIndex2 = tex1Dfetch(tex_secondIndex,flowNumber);
				float theta =   tex1Dfetch(tex_theta,flowNumber);

				int i = tex1Dfetch(tex_genIndex,flowNumber);

				int dataOffsetDiscounts = pathNumber+i*paths*numberDfsPerStepPerPath; 


				// discount ratio to start of step i
				float rdfToNow = tex1Dfetch(tex_discounts,dataOffsetDiscounts+i*paths);

				float rdfTo1 = tex1Dfetch(tex_discounts,dataOffsetDiscounts+dfIndex1*paths);
				float rdfTo2 = tex1Dfetch(tex_discounts,dataOffsetDiscounts+dfIndex2*paths);
				rdfTo1 /= rdfToNow;
				rdfTo2 /= rdfToNow;

				// ratio ratio to cash flows time 
				float rdf = pow(rdfTo1, 1.0f-theta)*pow(rdfTo2, theta);

				// df ratio from cash-flow time to step time 
				float df = rdf/rdfToNow;

				float flow = genFlows[dataOffset]*df;


				// now need to discount from step time to exercise time using the numeraire values 

				int targetIndex = tex1Dfetch(tex_targetIndex,flowNumber);

				flow *= numeraireValues[pathNumber+targetIndex*paths]/numeraireValues[pathNumber+i*paths];


				discountedFlows[dataOffset] = flow;

				dataOffset+= paths;



			}

		}
	}
}
extern"C"
void cashFlowDiscounting_partial_gpu(int numberFlowsPerPath,
									 int* genIndex_dev,
									 int* firstIndex_dev, 
									 int* secondIndex_dev,
									 float* theta_dev, 
									 int* targetIndices_dev,
									 float* discountRatios_dev, 
									 float* genFlows_dev, 
									 float* numeraireValues_dev,
									 int paths, 
									 int numberSteps, 
									 float* discountedFlows_dev) // output
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

	cudaTextureFloatBinder texBinder(tex_theta, theta_dev);
	cudaTextureFloatBinder texDiscountsBinder(tex_discounts, discountRatios_dev);

	cudaTextureIntBinder firstBinder(tex_firstIndex,firstIndex_dev);
	cudaTextureIntBinder secondBinder(tex_secondIndex,secondIndex_dev);

	cudaTextureIntBinder targetBinder(tex_targetIndex,targetIndices_dev);
	cudaTextureIntBinder genBinder(tex_genIndex,genIndex_dev);

	int numberDfs = numberSteps+1;
    
    ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);

	cashFlowDiscounting_partial_kernel<<<dimGrid , dimBlock >>>(numberFlowsPerPath,
		genFlows_dev, 
		numeraireValues_dev,
		paths,
		numberSteps, 
		numberDfs, 
		discountedFlows_dev);

}
