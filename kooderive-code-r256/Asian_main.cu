// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3
// note derived from code provided by NVIDIA 

// NVIDIA licensing terms: 
//Source Code: Developer shall have the right to modify and create derivative works with the Source Code.
//Developer shall own any derivative works ("Derivatives") it creates to the Source Code, provided that
//Developer uses the Materials in accordance with the terms and conditions of this Agreement. Developer
//may distribute the Derivatives, provided that all NVIDIA copyright notices and trademarks are used properly
//and the Derivatives include the following statement: "This software contains source code provided by
//NVIDIA Corporation."



#include "Asian.h"
#include <iostream>
#include <cutil_inline.h>
#include <math.h>
#include <thrust/version.h>
#include <vector>
#include <thrust/host_vector.h>
#include "Asian_gpu.h"







int AsianMainDevice(thrust::device_vector<float>& input_normals,
                                                                                                 int normalsDimensions, // must be greater than or equal to stepsPerPath
                                                                                                 int totalPaths, 
                                                                                                 int stepsPerPath,
                                                                                                 const std::vector<float>& logDrifts_vec, 
                                                                                                 const std::vector<float>& logSds_vec, 
                                                                                                 float logSpot0,
                                                                                                 float df,
                                                                                                 float strikeArithmetic,
                                                                                                 thrust::device_vector<float>& outputDataArithmetic,
                                                                                                 float strikeGeometric,
                                                                                                 thrust::device_vector<float>&outputDataGeometric)
{
 
    // Create a timer to measure performance
   Timer h1;

    int dataSize = normalsDimensions*totalPaths;

    if (input_normals.size() != dataSize)
        throw("input_normals must be of size  normalsDimensions*totalPaths it isn't.");

    outputDataArithmetic.resize(totalPaths);

        
    thrust::host_vector<float> logDrifts_host(logDrifts_vec.begin(),logDrifts_vec.end()); 
    thrust::host_vector<float> logSds_host(logSds_vec.begin(),logSds_vec.end()); 

                                                                                           
    cutilSafeCall(cudaThreadSynchronize());

    // Execute the QRNG on the device
    std::cout << "Executing Asian on GPU..." << std::endl;
   

    AsianCallGPU_kernel_dev(input_normals,
                                                                                                 normalsDimensions, // must be greater than or equal to stepsPerPath
                                                                                                 totalPaths, 
                                                                                                 stepsPerPath,
                                                                                                 logDrifts_host, 
                                                                                                 logSds_host, 
                                                                                                 logSpot0,
                                                                                                 df,
                                                                                                 strikeArithmetic,
                                                                                                 outputDataArithmetic,
                                                                                                 strikeGeometric,
                                                                                                 outputDataGeometric);

    cutilSafeCall(cudaThreadSynchronize());
    double time = h1.timePassed();
   
    std::cout << " time taken for Asian paths: " 
            << time << std::endl;
    
      return 0;
}

