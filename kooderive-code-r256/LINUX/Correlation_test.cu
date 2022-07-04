
//
//                                                                                                                                       Correlation_Test.cu
//
//
// (c) Mark Joshi 2009,2014
// This code is released under the GNU public licence version 3

#include <Correlation_test.h>
#include "Correlation_main.h"
#include <gold/Correlation_gold.h>
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
#include <inverseCumulatives.h>
#include <gold/Bridge_gold.h>
#include "sobol.h"

#include "Bridge.h"
#include "cudaMacros.h"

#include "Utilities.h"
// compare GPU against CPU

namespace
{
    double tolerance = 1E-6;
}

int CorrelationTestRoutine(bool verbose, bool useTextures, DeviceChooser& chooser)
{
    int path_offset=0;
    int result =1;
    {
        if (verbose)
        {
            std::cout << "\n\nTesting correlation routine ";
            if (!useTextures)
              std::cout << " not ";
            std::cout << " using textures.\n";
        }
         bool extraVerbose = false;
     
         cudaSetDevice(chooser.WhichDevice());
   

        int n_vectors =  intPower(2,20);
        int n_poweroftwo =5;
        int factors = 3;
        int n_steps = intPower(2,n_poweroftwo);
        int tot_dimensions = n_steps*factors;
        int N= n_vectors*tot_dimensions; 
        int number_rates = 5;
        
        
        size_t outN = n_vectors*n_steps*number_rates;
        
        
         size_t estimatedFloatsRequired = 4*(2*outN+2*N);
   
         size_t estimatedMem = sizeof(float)*estimatedFloatsRequired;

 //  bool change=false;
  
       ConfigCheckForGPU checker;

       float excessMem=2.0f;

       while (excessMem >0.0f)
       {
           checker.checkGlobalMem( estimatedMem, excessMem );

           if (excessMem >0.0f)
           {
               n_vectors/=2;
               N= n_vectors*tot_dimensions;
               outN = n_vectors*n_steps*number_rates;
  //             change =true;
               estimatedFloatsRequired = 4*(2*outN+2*N);
               estimatedMem = sizeof(float)*estimatedFloatsRequired;
     
           }
       }

        std::vector<float> A(number_rates*factors);
        
        for (int i=0; i < static_cast<int>(A.size()); ++i)
            A[i] = static_cast<float>(i+1.0f);
            
            
            
        thrust::host_vector<float> A_host(A.begin(),A.end());
        thrust::device_vector<float> A_device(A_host);
        std::vector<int> alive(n_steps);
        std::fill(alive.begin(),alive.end(),0);
         thrust::host_vector<int> alive_host(alive.begin(),alive.end());
        thrust::device_vector<int> alive_device(alive_host);





        cudaThreadSynchronize();
       
        thrust::device_vector<float> dev_output(N);    
        thrust::device_vector<float> device_input(N);
        
        bool doNormalInSobol =true;

         SobDevice( n_vectors, tot_dimensions, path_offset,device_input,doNormalInSobol);
     
        cudaThreadSynchronize();
    
      
        if (extraVerbose)
        {
              thrust::host_vector<float> bridgedNumbersHost(N);   
              bridgedNumbersHost = device_input;
            
              for (int i=0; i < N; ++i)
                 std::cout << i << "," << bridgedNumbersHost[i]  << "\n";
            
        }


      BrownianBridgeMultiDim<float>::ordering allocator( BrownianBridgeMultiDim<float>::triangular);


      MultiDBridge(n_vectors, 
                                     n_poweroftwo,
                                     factors,
                                     device_input, 
                                     dev_output,
                                     allocator,
                                     useTextures)
                                    ;

            	 
     
     thrust::host_vector<float> bridgedNumbersHost(N);   
     bridgedNumbersHost = dev_output;
     
      thrust::device_vector<float> dev_correlated_rates(outN);
      
      std::cout << "n_steps " << n_steps << "\n";
      
     Timer h1;
  
     
      
        correlated_paths_main_multistep_device(dev_output, // randon numbers
                                                                                                dev_correlated_rates, // correlated rate increments 
                                                                                                A_device, // correlator 
                                                                                                alive_device,
                                                                                                0,
                                                                                                factors, 
                                                                                                number_rates,
                                                                                                n_vectors,
                                                                                                n_steps);
                                                                                                
      
	double time = h1.timePassed();
    std::cout << " time taken for all steps of path correlations."    << time << std::endl;

    thrust::host_vector<float> host_correlated_rates(dev_correlated_rates);                                                                         
                                                                                                
                                                                                                
      std::vector<float> uncorrelatedNumbersVec(bridgedNumbersHost.begin(),bridgedNumbersHost.end());
       std::vector<float> correlatedNumbersVec(outN);
      
      
      int t0=clock();
      
      for (int i=0; i < n_steps; ++i)
            correlated_paths_gold(uncorrelatedNumbersVec, // randon numbers
                                                                        correlatedNumbersVec,  // correlated rate increments 
                                                                        A, // correlator 
                                                                        factors, 
                                                                        number_rates,
                                                                         n_vectors,
                                                                         i);

    int t1=clock();
    
    double time2 = (t1-t0+0.0)/CLOCKS_PER_SEC;
    
    std::cout << " time taken for CPU correlation " << time2 << "\n";
      
     std::cout << " speed up ratio " << time2/time << "\n";
     
     
  
     double err1=0.0;
     double err2 = 0.0;
     
     if (extraVerbose)
        for (int p = 0; p < n_vectors; ++p)
            for (int s=0; s < n_steps; ++s)
            {
                std::cout << p << ","<< s << "," ;
                for (int f=0; f < factors; ++f)     
            {  
                int i = p*factors+    n_vectors*s*factors+f;
                std::cout << uncorrelatedNumbersVec[i] << ",";
            
            }
            std::cout << "\n";
            }
     
     for (int p = 0; p < n_vectors; ++p)
        for (int s=0; s < n_steps; ++s)
        {
                if (extraVerbose)
                    std::cout << p << "," << s ;
      
            for (int r=0; r < number_rates; ++r)     
            {
                int i = p +    n_vectors*(r+ s*number_rates);
                if (extraVerbose)
                    std::cout <<  "," << host_correlated_rates[i] ;
                double erri = correlatedNumbersVec[i] - host_correlated_rates[i];
                err1 += fabs(erri);
                err2 += fabs(erri*erri);
            }
            if (extraVerbose)
                std::cout << "\n";
        }
                                                                                                

    double L1err = err1/(outN-number_rates*n_steps);
    
    double L2err = sqrt(err2/(outN-number_rates*n_steps));

      
    if (verbose)
    {
        std::cout << "L1  error " << L1err << "\n";
        std::cout << "L2  error " << L2err << "\n";
    }
      
    if (L1err > tolerance)
    {
        std::cout << " correlation  test failed";
    }
    else
        result =0;
    }
    
    cudaThreadExit();
      
 
    return result;
}
