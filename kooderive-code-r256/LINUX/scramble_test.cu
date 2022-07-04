//
//
//                                                                                                                                scramble_test.cu
//
//
// (c) Mark Joshi 2010,2014
// This code is released under the GNU public licence version 3

#include "scramble_test.h"
#include <gold/scramble_gold.h>
#include "scramble_main.h"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cutil_inline.h>
#include <cuda_runtime.h>
#include "cudaMacros.h"
#include <gold/Bridge_gold.h>
#include <cutil.h>

#include "Utilities.h"
#include <gold/MatrixFacade.h> 


// compare GPU against CPU



int scrambleTestRoutine(bool verbose,DeviceChooser& chooser)
{
    int retValue =0;

{
    if (verbose)
        std::cout << "\nTesting scrambling...\n";

    int n_vectors =  intPower(2,21);
    int n_poweroftwo =7;
    int n_dimensions = intPower(2,n_poweroftwo);
 
    size_t N= n_vectors*n_dimensions; 
    
          size_t estimatedFloatsRequired = 8*N;
   
         size_t estimatedMem = sizeof(float)*estimatedFloatsRequired;

       cudaSetDevice(chooser.WhichDevice());

 
  
       ConfigCheckForGPU checker;

       float excessMem=2.0f;

       while (excessMem >0.0f)
       {
           checker.checkGlobalMem( estimatedMem, excessMem );

           if (excessMem >0.0f)
           {
               n_vectors/=2;
               N =  n_vectors*n_dimensions;
               estimatedFloatsRequired = 8*N;
               estimatedMem = sizeof(float)*estimatedFloatsRequired;
     
           }
       }


    std::vector<unsigned int> scrambler_vec(n_dimensions);
    
    for (int i=0 ; i < n_dimensions; ++i)
        scrambler_vec[i] = 6*i*i + 2*i + 1;
        
     thrust::host_vector<unsigned int> scrambler_host(scrambler_vec.begin(),scrambler_vec.end() );
     
        
    std::vector<unsigned int> data_vec(N);
    MatrixFacade<unsigned int> data_matrix(&data_vec[0], n_dimensions, n_vectors);
    
    for (int i=0; i < n_dimensions; ++i)
        for (int p=0; p < n_vectors; ++p)
            data_matrix(i,p) = i*n_vectors+p;
        
    thrust::host_vector<unsigned int> data_host(data_vec.begin(),data_vec.end());
    
    std::vector<unsigned int> data_out_vec(N);
    
    if (verbose)
    {
        std::cout << "Entering scramber gold \n";
        std:: cout << " dimensions " << n_dimensions << "      number of vectors " << n_vectors << "\n";
     }
    
    int t0=clock();
    
     scramble_gold(data_vec, // data to scramble
                                       data_out_vec, //  scrambled data
                                       scrambler_vec,
                                       n_dimensions,
                                       n_vectors);
    
    int t1=clock();
    
    if (verbose)
        std::cout << "CPU time taken: " << (t1-t0+0.0)/CLOCKS_PER_SEC<< "\n";

     cudaThreadSynchronize();
       
    thrust::device_vector<unsigned int> data_device(data_host.begin(),data_host.end());
    thrust::device_vector<unsigned int> scrambler_device(scrambler_host.begin(),scrambler_host.end());
     
     cutilSafeCall(cudaThreadSynchronize());
     
     int t2=clock();
     
     scramble_main(data_device, // random numbers
                                                                                                scrambler_device,
                                                                                                n_dimensions,
                                                                                                 n_vectors);
   
      cudaThreadSynchronize();
      
      int t3=clock();
      
      if (verbose)
          std::cout << "GPU time taken: " << (t3-t2+0.0)/CLOCKS_PER_SEC<< "\n";
                  
       thrust::host_vector<int> out_host(data_device);
       
       int errs=0;
       
       for (int i=0; i < N;++i)
                if (out_host[i] != data_out_vec[i])
                   ++errs;
      
      
    if (errs > 0)
    {
        std::cout << " scramble test failed: " << errs << "   times";
        retValue =1;
    }
    }
    
    cudaThreadExit();
      
 
    return retValue;
}
