/*
* Copyright (c) 2010 Oh Kang Kwon. Released under GNU Public Licence Version 3.
Copyright (c) 2010 Mark Joshi. Released under GNU Public Licence Version 3.

*/

// Non CUDA specific code borrowed from quantlib...
//
// Copyright (C) 2003 Ferdinando Ametrano
// Copyright (C) 2006 StatPro Italia srl
// 
// This file is part of QuantLib, a free-software/open-source library
// for financial quantitative analysts and developers - http://quantlib.org/

// QuantLib is free software: you can redistribute it and/or modify it
// under the terms of the QuantLib license.  You should have received a
// copy of the license along with this program; if not, please email
// <quantlib-dev@lists.sf.net>. The license is also available online at
// <http://quantlib.org/license.shtml>.
// 
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the license for more details.

// ===========================================================================
// NOTE: The following copyright notice applies to the original code,
//
// Copyright (C) 2002 Peter J‰ckel "Monte Carlo Methods in Finance".
// All rights reserved.
//
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// ===========================================================================

#include "brownian_bridge.h"
#include <cuda.h>
#include <vector>
#include <cutil.h>

#include <cutil_inline.h>
#include <cuda_runtime.h>

#include <cudaWrappers/cudaTextureBinder.h>
#include "Utilities.h"
#include <gpuCompatibilityCheck.h>


//texture<float, 1, cudaReadModeElementType> tex_input;
//texture<float, 1, cudaReadModeElementType> tex_leftWeight;
//texture<float, 1, cudaReadModeElementType> tex_rightWeight;
//texture<float, 1, cudaReadModeElementType> tex_stdDev;

/*
texture<int, 1, cudaReadModeElementType> tex_indices;
texture<int, 1, cudaReadModeElementType> tex_bridgeIndex;
texture<int, 1, cudaReadModeElementType> tex_leftIndex;
texture<int, 1, cudaReadModeElementType> tex_rightIndex;
// There is an implicit assumption here that times are unit spaced.
*/

/*
__global__ void brownian_bridge_kernel(int paths, 
int steps, 
int factors,
int index_offset,
float* d_output
)
{
// First determine the path index to bridge
int path0 = threadIdx.x + blockIdx.x * blockDim.x;
// Next determine the factor index and offset into indices for the factor
int factor = blockIdx.y;

index_offset+= factor * steps;
// Stride
int stride = gridDim.x * blockDim.x;
// Bridging
int indexHelper;
for (int path = path0; path < paths; path += stride)
{
d_output[paths * (factors * (steps - 1) + factor) + path] =
tex1Dfetch(tex_stdDev,0)* tex1Dfetch(tex_input,paths *tex1Dfetch(tex_indices, index_offset)+ path);
for (int i = 1; i < steps; ++i) 
{
int j =tex1Dfetch(tex_leftIndex,i); 
int k =tex1Dfetch(tex_rightIndex,i); 
int l = tex1Dfetch(tex_bridgeIndex,i); 
if (j != 0) 
{
d_output[paths * (factors * l + factor) + path] 
= 
tex1Dfetch(tex_leftWeight,i)
* d_output[paths * (factors * (j - 1) + factor) + path] 
+ 
tex1Dfetch(tex_rightWeight,i)* d_output[paths * (factors * k + factor) + path] 
+ 
tex1Dfetch(tex_stdDev,i)
* tex1Dfetch(tex_input, paths * tex1Dfetch(tex_indices, index_offset+i) + path);

} 
else 
{
d_output[paths * (factors * l + factor) + path] 
=
tex1Dfetch(tex_rightWeight,i)* d_output[paths * (factors * k + factor) + path] 
+
tex1Dfetch(tex_stdDev,i)
*  tex1Dfetch(tex_input, paths *  tex1Dfetch(tex_indices, index_offset+i)+ path);
}
}
// ...after which, we calculate the differences
for (int i = steps - 1; i >= 1; --i) 
{
indexHelper = paths * (factors * i + factor) + path;
d_output[indexHelper] -= d_output[paths * (factors * (i - 1) + factor) + path];
}

}
}
*/
__global__ void brownian_bridge_kernel_fewer_textures(int paths, 
                                                      int steps, 
                                                      int factors,
                                                      int index_offset,
                                                      float* stdev_global,
                                                      const float* __restrict__ d_input,
                                                      float* l_weight_global,
                                                      float* r_weight_global,
                                                      int* leftIndex_global,
                                                      int* rightIndex_global,
                                                      int* bridgeIndex_global,
                                                      int* indices_global,
                                                      float* d_output
                                                      )
{
    extern __shared__ float stdevandweights_shared[];

    //    float* stdev_s = stdev_global; //
    float* stdev_s = stdevandweights_shared;

    //   float* weight_l_s = l_weight_global; 
    float* weight_l_s = stdevandweights_shared+steps;


    //     float* weight_r_s = r_weight_global; 

    float* weight_r_s = weight_l_s+steps;

    int* leftIndex_s = reinterpret_cast<int*>(weight_r_s+steps);
    int* rightIndex_s = leftIndex_s+steps;
    int* bridgeIndex_s = rightIndex_s+steps;
    int* indices_s = bridgeIndex_s+steps;

    int tx = threadIdx.x;

    while (tx < steps)
    {
        stdev_s[tx] = stdev_global[tx];
        weight_l_s[tx] = l_weight_global[tx];
        weight_r_s[tx] = r_weight_global[tx];
        leftIndex_s[tx] = leftIndex_global[tx];
        rightIndex_s[tx] = rightIndex_global[tx];
        bridgeIndex_s[tx] = bridgeIndex_global[tx];
        indices_s[tx] = indices_global[tx];
        tx += blockDim.x;
    }
    __syncthreads();


    // First determine the path index to bridge
    int path0 = threadIdx.x + blockIdx.x * blockDim.x;
    // Next determine the factor index and offset into indices for the factor
    int factor = blockIdx.y;

    index_offset+= factor * steps;
    // Stride
    int stride = gridDim.x * blockDim.x;
    // Bridging
    int indexHelper;
    for (int path = path0; path < paths; path += stride)
    {
        d_output[paths * (factors * (steps - 1) + factor) + path] =
            stdev_s[0]* d_input[paths *indices_global[index_offset]
        //tex1Dfetch(tex_indices, index_offset)
        + path];
        //tex1Dfetch(tex_input,paths *tex1Dfetch(tex_indices, index_offset)+ path);
        for (int i = 1; i < steps; ++i) 
        {
            int j =leftIndex_s[i];
            //tex1Dfetch(tex_leftIndex,i); 
            int k =rightIndex_global[i];
            //tex1Dfetch(tex_rightIndex,i); 
            int l =  bridgeIndex_global[i];
            //tex1Dfetch(tex_bridgeIndex,i); 
            if (j != 0) 
            {
                d_output[paths * (factors * l + factor) + path] 
                = 
                    // tex1Dfetch(tex_leftWeight,i)
                    weight_l_s[i]
                * d_output[paths * (factors * (j - 1) + factor) + path] 
                + 
                    //       tex1Dfetch(tex_rightWeight,i)* 
                    weight_r_s[i]
                *  d_output[paths * (factors * k + factor) + path] 
                + 
                    stdev_s[i]
                *d_input[paths* indices_global[index_offset+i]
                //tex1Dfetch(tex_indices, index_offset+i) 

                + path]; 
                //        tex1Dfetch(tex_input, paths * tex1Dfetch(tex_indices, index_offset+i) + path);

            } 
            else 
            {
                d_output[paths * (factors * l + factor) + path] 
                =
                    //tex1Dfetch(tex_rightWeight,i)* 
                    weight_r_s[i]
                *  d_output[paths * (factors * k + factor) + path] 
                +
                    stdev_s[i]
                *d_input[paths *  indices_global[index_offset+i]
                //tex1Dfetch(tex_indices, index_offset+i)
                + path]; 
                //     *  tex1Dfetch(tex_input, paths *  tex1Dfetch(tex_indices, index_offset+i)+ path);
            }
        }
        // ...after which, we calculate the differences
        for (int i = steps - 1; i >= 1; --i) 
        {
            indexHelper = paths * (factors * i + factor) + path;
            d_output[indexHelper] -= d_output[paths * (factors * (i - 1) + factor) + path];
        }

    }
}

/*
__global__ void brownian_bridge_kernel_wo_differencing(int paths, 
int steps, 
int factors,
int index_offset,
float* d_output
)
{
// First determine the path index to bridge
int path0 = threadIdx.x + blockIdx.x * blockDim.x;
// Next determine the factor index and offset into indices for the factor
int factor = blockIdx.y;

index_offset+= factor * steps;
// Stride
int stride = gridDim.x * blockDim.x;
int fs = factors * (steps - 1) ;
// Bridging

for (int path = path0; path < paths; path += stride)
{
d_output[paths * (fs+ factor) + path] =
tex1Dfetch(tex_stdDev,0)* tex1Dfetch(tex_input,paths *tex1Dfetch(tex_indices, index_offset)+ path);
for (int i = 1; i < steps; ++i) 
{
int j =tex1Dfetch(tex_leftIndex,i); 
int k =tex1Dfetch(tex_rightIndex,i); 
int l = tex1Dfetch(tex_bridgeIndex,i); 
if (j != 0) 
{
d_output[paths * (factors * l + factor) + path] 
= 
tex1Dfetch(tex_leftWeight,i)
* d_output[paths * (factors * (j - 1) + factor) + path] 
+ 
tex1Dfetch(tex_rightWeight,i)* d_output[paths * (factors * k + factor) + path] 
+ 
tex1Dfetch(tex_stdDev,i)
* tex1Dfetch(tex_input, paths * tex1Dfetch(tex_indices, index_offset+i) + path);

} 
else 
{
d_output[paths * (factors * l + factor) + path] 
=
tex1Dfetch(tex_rightWeight,i)* d_output[paths * (factors * k + factor) + path] 
+
tex1Dfetch(tex_stdDev,i)
*  tex1Dfetch(tex_input, paths *  tex1Dfetch(tex_indices, index_offset+i)+ path);
}
}

}
}

__global__ void brownian_bridge_kernel_wo_differencing_sf(int paths, 
int steps, 
int index_offset,
float* d_output
)
{
// First determine the path index to bridge
int path0 = threadIdx.x + blockIdx.x * blockDim.x;
// Next determine the factor index and offset into indices for the factor

// Stride
int stride = gridDim.x * blockDim.x;
int fs =  steps - 1 ;
// Bridging

for (int path = path0; path < paths; path += stride)
{
d_output[paths * fs + path] =
tex1Dfetch(tex_stdDev,0)* tex1Dfetch(tex_input,paths *tex1Dfetch(tex_indices, index_offset)+ path);
for (int i = 1; i < steps; ++i) 
{
int j =tex1Dfetch(tex_leftIndex,i); 
int k =tex1Dfetch(tex_rightIndex,i); 
int l = tex1Dfetch(tex_bridgeIndex,i); 
if (j != 0) 
{
d_output[paths * l + path] 
= 
tex1Dfetch(tex_leftWeight,i)
* d_output[paths * (j - 1) + path] 
+ 
tex1Dfetch(tex_rightWeight,i)* d_output[paths * k + path] 
+ 
tex1Dfetch(tex_stdDev,i)
* tex1Dfetch(tex_input, paths * tex1Dfetch(tex_indices, index_offset+i) + path);

} 
else 
{
d_output[paths * l+ path] 
=
tex1Dfetch(tex_rightWeight,i)* d_output[paths * k + path] 
+
tex1Dfetch(tex_stdDev,i)
*  tex1Dfetch(tex_input, paths *  tex1Dfetch(tex_indices, index_offset+i)+ path);
}
}

}
}
*/

// This implementation has output values in the following form
// (p0, s0, f0),   (p1, s0, f0),   ..., (pn_p, s0, f0),
// (p0, s0, f1),   (p1, s0, f1),   ..., (pn_p, s0, f1),
// ...............................................,
// (p0, s0, fn_f), (p1, s0, fn_f), ..., (pn_p, s0, fn_f),
// ...............................................
// That is in pathwise, factorwise, stepwise order.

brownian_bridge::brownian_bridge(int size, const std::vector<int>& allocation_indices)
: size_(size), t_(size), sqrtdt_(size), bridgeIndex_(size), leftIndex_(size)
, rightIndex_(size), leftWeight_(size), rightWeight_(size), stdDev_(size) , allocation_indices_(allocation_indices)
{
    for (int i = 0; i < size_; ++i)
    {
        t_[i] = static_cast<float>(i + 1);
    }

    deviceInitted_=false;

    initialize();
}

void brownian_bridge::transform(thrust::device_vector<float>& dev_input, 
                                thrust::device_vector<float>& dev_output,
                                int paths, 
                                int steps, 
                                int factors)
{
    initDevice();

    brownian_bridge::transformDevice(thrust::raw_pointer_cast(&dev_input[0]), thrust::raw_pointer_cast(&dev_output[0])
        ,  paths, steps, factors);

}



void brownian_bridge::transform(std::vector<float>& input_vec,  
                                std::vector<float>& output_vec,  
                                int paths, 
                                int steps, 
                                int factors)
{
    transformCPU(&input_vec[0], &output_vec[0],   paths, steps, factors);
}

void brownian_bridge::transform(float* input, float* output,  int paths, int steps, int factors, bool useDevice)
{
    if (useDevice)
        transformDevice( input, output,  paths, steps, factors);
    else
        transformCPU( input,  output,  paths, steps, factors);
}

void brownian_bridge::transformCPU(float* input_cpu, float* output_cpu,  int paths, int steps, int factors)
{
    MatrixFacade<int> indices(&allocation_indices_[0],factors,steps);
    CubeFacade<float> output(output_cpu,steps,factors,paths);
    MatrixConstFacade<float> input(input_cpu,factors*steps,paths);

    bool doDifferencing = true;


    for (int path = 0; path < paths; ++path )
    {
        for (int factor=0; factor<factors; ++factor)
        {
            output(steps-1,factor,path) = stdDev_[0]*input(indices(factor,0),path);

            for (int i = 1; i < steps; ++i) 
            {
                int j = leftIndex_[i]; 
                int k =rightIndex_[i]; 
                int l =bridgeIndex_[i];

                if (j != 0) 
                {
                    output(l,factor,path)  
                        = 
                        leftWeight_[i]
                    * output(j-1,factor,path)
                        + 
                        rightWeight_[i]* output(k,factor,path)
                        + 
                        stdDev_[i]
                    * input(indices(factor,i),path);


                } 
                else 
                {
                    output(l,factor,path)  
                        = 
                        rightWeight_[i]* output(k,factor,path)
                        +  stdDev_[i]
                    * input(indices(factor,i), path);
                }
            }
            // ...after which, we calculate the differences

            if (doDifferencing)
                for (int i = steps - 1; i >= 1; --i) 
                {
                    output(i,factor,path) -=  output(i-1,factor,path);
                }
        }

    }
}


void brownian_bridge::initDevice()
{
    if (deviceInitted_)
        return;

    t_dev_ = deviceVecFromStlVec(t_);
    sqrtdt_dev_ = deviceVecFromStlVec(sqrtdt_);
    bridgeIndex_dev_ = deviceVecFromStlVec(bridgeIndex_);
    leftIndex_dev_ = deviceVecFromStlVec(leftIndex_);
    rightIndex_dev_ = deviceVecFromStlVec(rightIndex_);
    leftWeight_dev_= deviceVecFromStlVec(leftWeight_);
    rightWeight_dev_= deviceVecFromStlVec(rightWeight_);
    stdDev_dev_= deviceVecFromStlVec(stdDev_);
    coefficient_dev_= deviceVecFromStlVec(coefficient_);
    indices_dev_= deviceVecFromStlVec(allocation_indices_);


    deviceInitted_=true;
}   



void brownian_bridge::transformDevice(float* d_input, float* d_output,  int paths, int steps, int factors)
{
    //    bool doDifferencing = true;
    //bool minimizeTextures = true;
    
    int size = 3*steps*sizeof(float)+4*steps*sizeof(int);

    compute_kernel_configuration(paths, factors,size);

    int* d_indices = thrust::raw_pointer_cast(&indices_dev_[0]);

    int index_offset =0;


    //    if (cudaSuccess!=cudaBindTexture(0, tex_indices, d_indices, sizeof(int)*indices_dev_.size()))
    //      GenerateError("Texture bind failed");


    float* lw_ptr_dev = thrust::raw_pointer_cast(&leftWeight_dev_[0]);

    float* rw_ptr_dev = thrust::raw_pointer_cast(&rightWeight_dev_[0]);


    float* st_ptr_dev = thrust::raw_pointer_cast(&stdDev_dev_[0]);
    //    if (cudaSuccess!=cudaBindTexture(0, tex_stdDev, st_ptr_dev, sizeof(float)*stdDev_dev_.size()))
    //      GenerateError("Texture bind failed");

    int* bi_ptr_dev = thrust::raw_pointer_cast(&bridgeIndex_dev_[0]);
    //  if (cudaSuccess!=cudaBindTexture(0, tex_bridgeIndex, bi_ptr_dev, sizeof(int)*bridgeIndex_dev_.size()))
    //    GenerateError("Texture bind failed");

    int* li_ptr_dev = thrust::raw_pointer_cast(&leftIndex_dev_[0]);
    //  if (cudaSuccess!=cudaBindTexture(0, tex_leftIndex, li_ptr_dev, sizeof(int)*leftIndex_dev_.size()))	
    //   GenerateError("Texture bind failed");

    int* ri_ptr_dev = thrust::raw_pointer_cast(&rightIndex_dev_[0]);
    //  if (cudaSuccess!=cudaBindTexture(0, tex_rightIndex, ri_ptr_dev, sizeof(int)*rightIndex_dev_.size()))
    //      GenerateError("Texture bind failed");


    /*
    if (!minimizeTextures)
    {
    if (cudaSuccess!=cudaBindTexture(0, tex_input, d_input, sizeof(float)*paths* steps*factors))
    GenerateError("Texture bind failed");
    if (cudaSuccess!=cudaBindTexture(0, tex_leftWeight, lw_ptr_dev, sizeof(float)*leftWeight_dev_.size()))
    GenerateError("Texture bind failed");
    if (cudaSuccess!=cudaBindTexture(0, tex_rightWeight, rw_ptr_dev, sizeof(float)*rightWeight_dev_.size()))
    GenerateError("Texture bind failed");
    }
    */

    cutilSafeCall(cudaThreadSynchronize());


  //  std::cout << " about to enter THE RELEVANT kernel ...";

  Timer h1;

    //  if (minimizeTextures)
    brownian_bridge_kernel_fewer_textures<<<dimGrid_, dimBlock_,size>>>(paths, 
        steps, 
        factors,
        index_offset,
        st_ptr_dev,
        d_input,
        lw_ptr_dev,
        rw_ptr_dev,
        li_ptr_dev,
        ri_ptr_dev,
        bi_ptr_dev,
        d_indices,
        d_output
        );

    /*  else
    if (doDifferencing)
    brownian_bridge_kernel<<<dimGrid_, dimBlock_>>>(paths, 
    steps, 
    factors,
    index_offset,
    d_output
    );
    else 
    if (factors >1)
    {        brownian_bridge_kernel_wo_differencing<<<dimGrid_, dimBlock_>>>(paths, 
    steps, 
    factors,
    index_offset,
    d_output
    );
    }
    else
    brownian_bridge_kernel_wo_differencing_sf<<<dimGrid_, dimBlock_>>>(paths, 
    steps, 
    index_offset,
    d_output
    );

    */
    cutilSafeCall(cudaThreadSynchronize());

    cudaThreadSynchronize();
    double  time = h1.timePassed();



   //  std::cout << "GPU time taken just for kernel call" << time*1000 << " ms\n";

    /*   if (!minimizeTextures) 
    {
    cudaUnbindTexture(tex_input);										  
    cudaUnbindTexture(tex_leftWeight);													
    cudaUnbindTexture(tex_rightWeight);																											  
    }
    */

    //    cudaUnbindTexture(tex_indices);
    //    cudaUnbindTexture(tex_bridgeIndex);		
    //    cudaUnbindTexture(tex_leftIndex);		
    //    cudaUnbindTexture(tex_rightIndex);	
    //    cudaUnbindTexture(tex_stdDev);													

}



void brownian_bridge::compute_kernel_configuration(int paths, int factors, int mem_size)
{	
    int const threadsperblock = 128;
    int device;
    cudaDeviceProp deviceproperty;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceproperty, device);          
    dimGrid_.y = factors;
    if (factors < 4 * deviceproperty.multiProcessorCount)
    {
        dimGrid_.x = 16 * 4 * deviceproperty.multiProcessorCount;
    }
    else
    {
        dimGrid_.x = 1;
    }
    if (dimGrid_.x > static_cast<unsigned int>(paths / threadsperblock))
    {
        dimGrid_.x = (paths + threadsperblock - 1)/ threadsperblock;
    }

    unsigned int targetDimGridx = dimGrid_.x;
    for (dimGrid_.x = 1; dimGrid_.x < targetDimGridx; dimGrid_.x <<= 1);
    dimBlock_.x = threadsperblock;

    if (deviceproperty.sharedMemPerBlock < mem_size)
        GenerateError("Too much shared memory required for Brownian bridge");
     ConfigCheckForGPU().checkConfig( dimBlock_,  dimGrid_);

}


void brownian_bridge::initialize()
{
    sqrtdt_[0] = std::sqrt(t_[0]);
    for (int i = 1; i < size_; ++i)
    {
        sqrtdt_[i] = std::sqrt(t_[i] - t_[i - 1]);
    }
    // map is used to indicate which points are already constructed.
    // If map[i] is zero, path point i is yet unconstructed.
    // map[i]-1 is the index of the variate that constructs
    // the path point # i.
    std::vector<int> map(size_, 0);
    //  The first point in the construction is the global step.
    map[size_ - 1] = 1;
    //  The global step is constructed from the first variate.
    bridgeIndex_[0] = size_ - 1;
    //  The variance of the global step
    stdDev_[0] = std::sqrt(t_[size_ - 1]);
    //  The global step to the last point in time is special.
    leftWeight_[0] = rightWeight_[0] = 0.0;
    for (int j = 0, i = 1; i < size_; ++i) 
    {
        // Find the next unpopulated entry in the map.
        while (map[j])
        {
            ++j;
        }
        int k = j;
        // Find the next populated entry in the map from there.
        while (!map[k])
        {
            ++k;
        }
        // l - 1 is now the index of the point to be constructed next.
        int l = j + ((k - 1 - j) >> 1);
        map[l] = i;
        // The i-th Gaussian variate will be used to set point l-1.
        bridgeIndex_[i] = l;
        leftIndex_[i]   = j;
        rightIndex_[i]  = k;
        if (j != 0) 
        {
            leftWeight_[i]= (t_[k] - t_[l]) / (t_[k] - t_[j - 1]);
            rightWeight_[i] = (t_[l] - t_[j - 1])/(t_[k] - t_[j - 1]);
            stdDev_[i] = std::sqrt(((t_[l] - t_[j - 1]) * (t_[k] - t_[l])) / (t_[k] - t_[j - 1]));
        } 
        else 
        {
            leftWeight_[i]  = (t_[k] - t_[l]) / t_[k];
            rightWeight_[i] = t_[l] / t_[k];
            stdDev_[i] = std::sqrt(t_[l] * (t_[k] - t_[l]) / t_[k]);
        }
        j = k + 1;
        if (j >= size_)
        {
            j = 0;    //  wrap around
        }
    }
}



