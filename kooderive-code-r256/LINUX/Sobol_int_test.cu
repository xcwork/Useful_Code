//
//
//                                                                                                                                          Sobol_int_Test.cu
//
//
// (c) Mark Joshi 2010,2014
// This code is released under the GNU public licence version 3

#include "Sobol_int_test.h"
#include <gold/pragmas.h> 
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <cutil.h>



#include "sobol.h"
#include "sobol_int_gpu.h"
#include <gold/sobol_gold.h>
#include "cudaMacros.h"
#include <gold/Bridge_gold.h>

#include "Utilities.h"

#include <gold/MatrixFacade.h> 

// compare GPU against CPU


int SobolTestRoutineForInts(bool verbose,DeviceChooser& chooser)
{
    std::cout << " Entering Sobol ints test... \n";

  double L1;

{
    bool extraVerbose = false;
     unsigned path_offset=0;


     int n_vectors =  intPower(2,23);
      
      
     int n_poweroftwo =5;
     int n_dimensions = intPower(2,n_poweroftwo);
 
     size_t N =  n_vectors*n_dimensions;
    
            
         size_t estimatedFloatsRequired = 8*N;
   
         size_t estimatedMem = sizeof(float)*estimatedFloatsRequired;

 //  bool change=false;
            
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

       int n_vectors1 = n_vectors/2;
     int n_vectors2 =n_vectors-n_vectors1;
  
        size_t N1= n_vectors1*n_dimensions; 
     size_t N2= n_vectors2*n_dimensions; 
 std::cout << "batch sizes " << n_vectors1 << " and " << n_vectors2 << "\n";


     cudaThreadSynchronize();
       
      thrust::device_vector<unsigned int> dev_output1(N1);    
      thrust::device_vector<unsigned int> dev_output2(N2);    

     
      cutilSafeCall(cudaThreadSynchronize());
   
      cudaThreadSynchronize();
       
      


       SobDeviceInts( n_vectors1, n_dimensions, path_offset, dev_output1);
     
       if (n_vectors2 > 0)
             SobDeviceInts( n_vectors2, n_dimensions, n_vectors1+
                                                    path_offset, dev_output2);
  
      cudaThreadSynchronize();

         
       thrust::host_vector<unsigned int> host_output1(dev_output1);
       thrust::host_vector<unsigned int> host_output2(dev_output2);
       
       std::vector<unsigned int> directions_vec(n_dimensions*n_directions);
       initSobolDirectionVectors(n_dimensions, &directions_vec[0]);
       std::vector<unsigned int> output_cpu_vec(n_vectors*n_dimensions);
       
       sobolCPUintsOffset(n_vectors,  n_dimensions,  0,&directions_vec[0], &output_cpu_vec[0]);
       
                                 
      double err=0.0;
      
      int start = 0;

     MatrixFacade<unsigned int> Sob_gpu_1(&host_output1[0],n_dimensions,n_vectors1);
     MatrixFacade<unsigned int> Sob_gpu_2(&host_output2[0],n_dimensions,n_vectors2);
     MatrixFacade<unsigned int> Sob_cpu(&output_cpu_vec[0],n_dimensions,n_vectors);
   
    int i=start;
     
     for ( ; i < n_vectors1; ++i)
     {
          if (extraVerbose)
                {
                    std::cout << "\n" << "," << i << ",";
                }
        for (int d=0; d < n_dimensions; ++d)
        {
                  unsigned int val_cpu =  Sob_cpu(d,i);
                  unsigned int val_gpu = Sob_gpu_1(d,i);
                
                if (extraVerbose)
                {
                    std::cout << val_cpu << "," << val_gpu << ",";
                }
         
            err+=    fabs(static_cast<double>( Sob_cpu(d,i) -  Sob_gpu_1(d,i)));           
         }
    }
    
    
     double err1 = err/N1;
     double err2=0.0;
    
      for ( ; i < n_vectors; ++i)
     {
         if (extraVerbose)
                {
                    std::cout << "\n" << "," << i << ",";
                }
        for (int d=0; d < n_dimensions; ++d)
        {
                unsigned int val_cpu =  Sob_cpu(d,i);
                unsigned int val_gpu = Sob_gpu_2(d,i-n_vectors1);
                
                if (extraVerbose)
                {
                    std::cout << val_cpu << "," << val_gpu << ",";
                }
         
            err2+=    fabs( val_cpu - val_gpu+0.0);          
         }
    }
    



    L1 = (err+err2)/N;
  
   if (n_vectors2>0)
     err2 /= N2;
      
    if (verbose)
    {
        std::cout << "L1  " << L1   << " first err: " << err1 << "          second error:" << err2 << "\n";
    }
    
    }
    
        
    cudaThreadExit();
      
    if (L1 > 0)
    {
        std::cout << " Sobol  int test failed";
        return 1;
    }
   
        
      
 
    return 0;
}


