//
//
//                                                                                                                                          Sobol_Test.cu
//
//
// (c) Mark Joshi 2010,2014
// This code is released under the GNU public licence version 3

#include "Sobol_test.h"
#include <gold/pragmas.h> 
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cutil_inline.h>
#include <cutil.h>

#include <cuda_runtime.h>
#include "InverseCumulatives.h"

#include "sobol.h"
#include "sobol_gpu.h"

#include "cudaMacros.h"
#include <gold/Bridge_gold.h>
#include <gold/sobol_gold.h>
#include "Utilities.h"

#include <gold/MatrixFacade.h> 

// compare GPU against CPU


namespace
{
    double tolerance = 5E-6;
}
int SobolTestRoutine(bool verbose,DeviceChooser& chooser)
{
    std::cout << " Entering Sobol test... \n";

  double L1;

{
    bool extraVerbose = false;
     unsigned path_offset=0;
     bool doNormals=false;

     int n_vectors =  intPower(2,16);
     int n_vectors1 = intPower(2,16);
     int n_vectors2 =n_vectors-n_vectors1;
     
     int n_poweroftwo =5;
     int n_dimensions = intPower(2,n_poweroftwo);
 
     int N =  n_vectors*n_dimensions;
     int N1= n_vectors1*n_dimensions; 
     int N2= n_vectors2*n_dimensions; 

     cudaSetDevice(chooser.WhichDevice());

     cudaThreadSynchronize();
       
      thrust::device_vector<float> dev_output1(N1);    
      thrust::device_vector<float> dev_output2(N2);    

     
      cutilSafeCall(cudaThreadSynchronize());
   
      cudaThreadSynchronize();
       
    ;
   
       int res1 =  SobDevice( n_vectors1, n_dimensions, path_offset, dev_output1,doNormals);
       if (n_vectors2 > 0)
             SobDevice( n_vectors2, n_dimensions, n_vectors1+
                                                    path_offset, dev_output2,doNormals);
  
      cudaThreadSynchronize();

         
       thrust::host_vector<float> host_output1(dev_output1);
       thrust::host_vector<float> host_output2(dev_output2);
       
       std::vector<unsigned int> directions_vec(n_dimensions*n_directions);
       initSobolDirectionVectors(n_dimensions, &directions_vec[0]);
       std::vector<float> output_cpu_vec(n_vectors*n_dimensions);
       
       sobolCPU(n_vectors,  n_dimensions,  &directions_vec[0], &output_cpu_vec[0]);
       
       inverseCumulative<float> inv;
       
       if (doNormals)
        std::transform(output_cpu_vec.begin(), output_cpu_vec.end(),output_cpu_vec.begin(),inv);
        
                                 
      double err=0.0;
      
      int start = 0;

     MatrixFacade<float> Sob_gpu_1(&host_output1[0],n_dimensions,n_vectors1);
     MatrixFacade<float> Sob_gpu_2(&host_output2[0],n_dimensions,n_vectors2);
     MatrixFacade<float> Sob_cpu(&output_cpu_vec[0],n_dimensions,n_vectors);
   
    int i=start;
     
     for ( ; i < n_vectors1; ++i)
     {
          if (extraVerbose)
                {
                    std::cout << "\n" << "," << i << ",";
                }
        for (int d=0; d < n_dimensions; ++d)
        {
                  float val_cpu =  Sob_cpu(d,i);
                  float val_gpu = Sob_gpu_1(d,i);
                
                if (extraVerbose)
                {
                    std::cout << val_cpu << "," << val_gpu << ",";
                }
         
            err+=    fabs( Sob_cpu(d,i) -  Sob_gpu_1(d,i));           
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
                float val_cpu =  Sob_cpu(d,i);
                float val_gpu = Sob_gpu_2(d,i-n_vectors1);
                
                if (extraVerbose)
                {
                    std::cout << val_cpu << "," << val_gpu << ",";
                }
         
            err2+=    fabs( val_cpu - val_gpu);          
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
      
    if (L1 > tolerance)
    {
        std::cout << " Sobol  test failed";
        return 1;
    }
   
        
      
 
    return 0;
}

