// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3


/*
* correlated variates

The objective is to multiply a large number of paths by the same matrix and add a fixed drift term to each . 

We should think
number of paths ~10.000                 N
number of factors ~ 5                         F            
number of correlated rates ~ 60  n

The input is a table of (quasi-)random numbers.  

Also need an input matrix of size n \times F for each step. 

The drift is a table of n times step.  

The output is a table of log stock or rate increments. All paths rate 0 then all paths rate 1 and so on... 

*/

#include "Correlate_drift_gpu.h"
#include "cudaMacros.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cutil.h>

#include <cutil_inline.h>


texture<float, 1, cudaReadModeElementType> tex_correlation;
texture<float, 1, cudaReadModeElementType> tex_A;

texture<int, 1, cudaReadModeElementType> tex_alive;


texture<float, 1, cudaReadModeElementType> tex_drift_add;

#define MAXCACHESIZE 3600

__global__ void correlation_drift_shared_kernel(             int input_dataOffset, // random numbers, offset for tex_correlation texture
                                                float* output_data, // correlated rate increments  plus drifts 
                                                int AOffset, // texture offset for first pseudo-root 
                                                int AOffsetPerStep,  // change in texture offset for each pseudo-root 
                                                int driftOffset, 
                                                int factors, 
                                                int out_dimensions,
                                                int number_paths,
                                                int alive_offset, // offset for tex_alive
                                                int steps
                                                )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;
    int inputLayerSize =factors*number_paths; 

    __shared__ float variateStore[MAXCACHESIZE];

    float* variateLoc = variateStore+tx*factors;


    int pathsPerThread = 1 + (( number_paths -1)/width);

    for (int l=0; l < pathsPerThread; ++l)
    {
        int pathNumber = width*l + bwidth*bx+tx;
        if (pathNumber < number_paths)
        {

            int input_data_this_step_offset =input_dataOffset+pathNumber*factors;
            float* first_output = output_data+pathNumber;

            int AOffset_this_step = AOffset;
            int driftOffset_this_step = driftOffset;

            for (int s=0; s <steps;++s)
            {
                int aliveIndex=tex1Dfetch(tex_alive,s+alive_offset);

                int roffset =AOffset_this_step+aliveIndex*factors; // first alive row of the pseudo-root matrix
                int drift_rowstart = driftOffset_this_step;// +aliveIndex*out_dimensions; // first alive row of the drift matrix 

                int outoffset =aliveIndex*number_paths;

                for (int f=0; f < factors; ++f)
                    variateLoc[f] =   tex1Dfetch(tex_correlation,input_data_this_step_offset+f);



                for (int r=aliveIndex; r  < out_dimensions; ++r)            
                {
                    float total =0.0;

                    for (int f=0; f < factors; ++f)
                        total +=  tex1Dfetch(tex_A,roffset+f)*   variateLoc[f] ;




                    int tex_drift_location = drift_rowstart+r;
                    float adder = tex1Dfetch(tex_drift_add,tex_drift_location);     

                    total+=adder;

                    //            incrementLoc[r*bwidth] = total;

                    first_output[outoffset] = total;


                    roffset +=    factors; // move pseudo-root to next row 
                    outoffset+= number_paths;

                } // end of rate loop


                input_data_this_step_offset+=   inputLayerSize;
                AOffset_this_step+= AOffsetPerStep;
                driftOffset_this_step += out_dimensions;
                first_output += out_dimensions*number_paths;



            } // end of steps lopp


        }// end of path valid test

    } // end of paths loop
}// end of function



__global__ void correlation_drift_kernel(          int input_dataOffset, // random numbers, offset for tex_correlation texture
                                         float* output_data, // correlated rate increments  plus drifts 
                                         int AOffset, // texture offset for first pseudo-root 
                                         int AOffsetPerStep,  // change in texture offset for each pseudo-root 
                                         int driftOffset, 
                                         int factors, 
                                         int out_dimensions,
                                         int number_paths,
                                         int alive_offset, // offset for tex_alive
                                         int steps
                                         )
{

    int bx = blockIdx.x;
    int tx =  threadIdx.x;


    int gx = gridDim.x;
    int bwidth = blockDim.x;
    int width = gx*bwidth;
    int inputLayerSize =factors*number_paths; 

    int pathsPerThread = 1 + (( number_paths -1)/width);

    int outLayerSize = out_dimensions*number_paths;

    for (int l=0; l < pathsPerThread; ++l )
    {
        int pathNumber = l*width+ bwidth*bx+tx;

        if (pathNumber < number_paths)
        {

            int input_data_this_step_offset =input_dataOffset+pathNumber*factors;
            float* first_output = output_data+pathNumber;

            int AOffset_this_step = AOffset;
            int driftOffset_this_step = driftOffset;

            for (int s=0; s <steps;++s)
            {
                int aliveIndex=tex1Dfetch(tex_alive,s+alive_offset);

                int roffset =AOffset_this_step+aliveIndex*factors; // first alive row of the pseudo-root matrix
                int drift_rowstart = driftOffset_this_step;  // first alive row of the drift matrix 

                int outoffset =aliveIndex*number_paths;


                for (int r=aliveIndex; r  < out_dimensions; ++r)            
                {
                    float total =0.0;

                    for (int f=0; f < factors; ++f)
                    {
                        total +=  tex1Dfetch(tex_A,roffset)* tex1Dfetch(tex_correlation,input_data_this_step_offset+f);

                        ++roffset;                                       
                    }                                                                          

                    total += tex1Dfetch(tex_drift_add,drift_rowstart+r);                      
                    first_output[outoffset] = total;                            
                    outoffset+= number_paths;

                }

                input_data_this_step_offset+=   inputLayerSize;
                AOffset_this_step+= AOffsetPerStep;
                driftOffset_this_step += out_dimensions;
                first_output += outLayerSize;

            }

        }

    }
}



extern "C"
void correlated_drift_paths_GPU(    float* input_data_device, // randon numbers
                                float* output_data_device, // correlated rate increments 
                                float* A_device, 
                                int* alive_device,
                                float* drift_fixed_device, 
                                int A_offset_per_step,
                                int factors, 
                                int out_dimensions,
                                int number_paths,
                                int steps)
{
     Timer h1;

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


    CUT_CHECK_ERR("correlated_paths_GPU execution failed before entering kernel\n");


    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    // set texture parameters
    tex_correlation.addressMode[0] = cudaAddressModeWrap;
    tex_correlation.addressMode[1] = cudaAddressModeWrap;
    tex_correlation.filterMode = cudaFilterModeLinear;
    tex_correlation.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTexture( NULL, tex_correlation, input_data_device, channelDesc);

    tex_A.addressMode[0] = cudaAddressModeWrap;
    tex_A.addressMode[1] = cudaAddressModeWrap;
    tex_A.filterMode = cudaFilterModeLinear;
    tex_A.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTexture( NULL, tex_A, A_device, channelDesc);

    tex_alive.addressMode[0] = cudaAddressModeWrap;
    tex_alive.addressMode[1] = cudaAddressModeWrap;
    tex_alive.filterMode = cudaFilterModeLinear;
    tex_alive.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTexture( NULL, tex_alive, alive_device, channelDesc);

    // set texture parameters
    tex_drift_add.addressMode[0] = cudaAddressModeWrap;
    tex_drift_add.addressMode[1] = cudaAddressModeWrap;
    tex_drift_add.filterMode = cudaFilterModeLinear;
    tex_drift_add.normalized = false;    // access with normalized texture coordinates

    // Bind the array to the texture
    cudaBindTexture( NULL, tex_drift_add, drift_fixed_device, channelDesc);


    CUT_CHECK_ERR("correlated_paths_multistep_GPU textures execution failed before entering kernel\n");

    int offset=0;

    bool useShared = false;

            
    ConfigCheckForGPU().checkConfig( dimBlock,  dimGrid);

    if (useShared)

        correlation_drift_shared_kernel<<<dimGrid , dimBlock >>>(offset, // random numbers, offset for texture 
        output_data_device, 
        offset, // texture offset for first pseudo-root 
        A_offset_per_step,
        offset, // offset for drift texture
        factors, 
        out_dimensions,
        number_paths,
        offset, // offset for tex_alive
        steps);  
    else  
        correlation_drift_kernel<<<dimGrid , dimBlock >>>(offset, // random numbers, offset for texture 
        output_data_device, 
        offset, // texture offset for first pseudo-root 
        A_offset_per_step,
        offset, // offset for drift texture
        factors, 
        out_dimensions,
        number_paths,
        offset, // offset for tex_alive
        steps);

    CUT_CHECK_ERR("correlation_kernel_texture_multistep execution failed after entering kernel\n");     
    cudaThreadSynchronize();                               

    cudaUnbindTexture(tex_correlation);

    cudaUnbindTexture(tex_A);
    cudaUnbindTexture(tex_drift_add);
    cudaUnbindTexture(tex_alive);


  //   std::cout << " time taken in correlation drift kernel " <<    h1.timePassed() << "\n";


}
