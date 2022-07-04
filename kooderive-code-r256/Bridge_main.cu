// (c) Mark Joshi 2009, 2010
// This code is released under the GNU public licence version 3

#include "Bridge.h"
#include <iostream>
#include <cutil_inline.h>
#include <math.h>
#include <thrust/version.h>
#include <vector>
#include <thrust/host_vector.h>
#include "Bridge_gpu.h"
#include "Bridge_multi_gpu.h"



int bridgeMainDevice(int n_vectors, 
                                int n_poweroftwo,
                                thrust::device_vector<float>& dev_inputGPU, 
                                thrust::device_vector<float>& dev_outputGPU,
                                bool report)
{
	int dim = static_cast<int>(pow(2.0, n_poweroftwo));

    int dataSize = dim*n_vectors;

    if (dev_inputGPU.size() != dataSize)
        throw("dev_inputGPU  must be of size  dim*n_vectorsand it isn't.");

    dev_outputGPU.resize(dataSize);

    float* d_input =   thrust::raw_pointer_cast(&*dev_inputGPU.begin());

    // we create an empty output vector on the device
    float* d_output =   thrust::raw_pointer_cast(&*dev_outputGPU.begin());

    cutilSafeCall(cudaThreadSynchronize());

    // Execute the QRNG on the device
     if (report)
            std::cout << "Executing bridge on GPU..." << std::endl;
   
    Timer h1;

    brownianBridgeGPU_constant_memory(n_vectors,  n_poweroftwo,d_input, d_output);

    cutilSafeCall(cudaThreadSynchronize());
    double time = h1.timePassed();
        
	if (report)
         std::cout << " time taken for bridge: " 
            << time  << std::endl;
    
  
    std::cout << std::endl;

    
    // Cleanup and terminate
    //std::cout << "Shutting down bridge..." << std::endl;
 ;
  
  //  cudaThreadExit();
    return 0;
}

int MultiDBridgeReordering(int n_vectors, 
                                int n_poweroftwo,
                                int factors,
                                thrust::device_vector<float>& dev_inputGPU, 
                                thrust::device_vector<float>& dev_outputGPU,
                                BrownianBridgeMultiDim<float>::ordering allocator,
                                bool useTextures)
{
     

    int dim = static_cast<int>(pow(2.0, n_poweroftwo));

    int dataSize = dim*n_vectors*factors;

    if (dev_inputGPU.size() != dataSize)
        throw("dev_inputGPU  must be of size  dim*n_vectors*factors and it isn't.");

    dev_outputGPU.resize(dataSize);

    float* d_input =   thrust::raw_pointer_cast(&*dev_inputGPU.begin());

    // we create an empty output vector on the device
    float* d_output =   thrust::raw_pointer_cast(&*dev_outputGPU.begin());

    cutilSafeCall(cudaThreadSynchronize());

//    std::cout << "      Executing reordering for bridge on GPU..." << std::endl;

    brownianBridgeMultiGPUReorder(d_input,
                                                                                                 d_output,
                                                                                                n_poweroftwo,
                                                                                                factors,
                                                                                                n_vectors,
                                                                                                  allocator,
                                                                                                  useTextures
                                                                                                 );

 

    return 0;
}





int MultiDBridge(int n_vectors, 
                                int n_poweroftwo,
                                int factors,
                                thrust::device_vector<float>& dev_inputGPU, 
                                thrust::device_vector<float>& dev_outputGPU,
                                BrownianBridgeMultiDim<float>::ordering allocator,
                                bool useTextures, 
                                bool report)
{
       // Create a timer to measure performance
  
    int dim = static_cast<int>(pow(2.0, n_poweroftwo));

    int dataSize = dim*n_vectors*factors;

    if (dev_inputGPU.size() != dataSize)
        throw("dev_inputGPU  must be of size  dim*n_vectors*factors and it isn't.");

    dev_outputGPU.resize(dataSize);

    float* d_input =   thrust::raw_pointer_cast(&*dev_inputGPU.begin());

    // we create an empty output vector on the device
    float* d_output =   thrust::raw_pointer_cast(&*dev_outputGPU.begin());

    cutilSafeCall(cudaThreadSynchronize());

    // Execute the QRNG on the device
   if (report)
       std::cout << "      Executing reordering for bridge on GPU..." << std::endl;

   Timer h1;

   brownianBridgeMultiGPUReorder(d_input,
                                                                                                 d_output,
                                                                                                n_poweroftwo,
                                                                                                factors,
                                                                                                n_vectors,
                                                                                                  allocator,
                                                                                                  useTextures
                                                                                                 );

      double time = h1.timePassed();
   
       if (report)
           std::cout << "       time taken for reordering: " 
            << time << std::endl;
    
    
   
    dev_inputGPU.swap(dev_outputGPU);

  // dev_inputGPU = dev_outputGPU;

    bridgeMainDevice(n_vectors*factors, 
                                n_poweroftwo,
                                dev_inputGPU, 
                                dev_outputGPU,
                                report);


    return 0;
}

