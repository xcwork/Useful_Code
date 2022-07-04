
//
//                                                                                                                                       Drift_add_Test.cu
//
//
// (c) Mark Joshi 2009,2013,2014
// This code is released under the GNU public licence version 3

#include "drift_add_test.h"
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
#include "Correlation_main.h"
#include "drift_add_main.h"
#include "drift_add_test.h"

float tolerance = 1E-6f;

// compare GPU against CPU
int DriftAddTestRoutine(bool verbose,  bool useTexture, DeviceChooser& chooser)
{
    int path_offset=0;
    int result =1;
    {
        if (verbose)
            std::cout << "\n\nTesting drift add  routine.\n";
            
         bool extraVerbose = false;
     
         cudaSetDevice(chooser.WhichDevice());
   

        int n_vectors =  intPower(2,20);
        int n_poweroftwo =5;
        int factors = 3;
        int n_steps = intPower(2,n_poweroftwo);
        int tot_dimensions = n_steps*factors;
        int N= n_vectors*tot_dimensions; 
        int number_rates = 30;
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
            
       std::vector<float> adder(number_rates*n_steps);
       
       for (int i=0; i < static_cast<int>(adder.size()); ++i)
            adder[i] =  static_cast<float>(i/10.0);
            
            
            
        thrust::host_vector<float> A_host(A.begin(),A.end());
        thrust::device_vector<float> A_device(A_host);

        thrust::host_vector<float> adder_host(adder.begin(),adder.end());
        thrust::device_vector<float> adder_device(adder_host);

        std::vector<int> alive(n_steps);
        std::fill(alive.begin(),alive.end(),0);
         thrust::host_vector<int> alive_host(alive.begin(),alive.end());
        thrust::device_vector<int> alive_device(alive_host);

        


        cudaThreadSynchronize();
       
        thrust::device_vector<float> dev_output(N);    
        thrust::device_vector<float> device_input(N);

        bool doNormalInSobol = true;

         SobDevice( n_vectors, tot_dimensions, path_offset, device_input,doNormalInSobol);
  
        cudaThreadSynchronize();
        
      BrownianBridgeMultiDim<float>::ordering allocator( BrownianBridgeMultiDim<float>::triangular);


      MultiDBridge(n_vectors, 
                                     n_poweroftwo,
                                     factors,
                                     device_input, 
                                     dev_output,
                                     allocator,
                                     useTexture)
                                    ;
   
      thrust::device_vector<float> dev_correlated_rates(outN);
      
     correlated_paths_main_multistep_device(dev_output, // randon numbers
                                                                                                dev_correlated_rates, // correlated rate increments 
                                                                                                A_device, // correlator 
                                                                                                alive_device,
                                                                                                0,
                                                                                                factors, 
                                                                                                number_rates,
                                                                                                n_vectors,
                                                                                                n_steps);
                                                                                                
    thrust::host_vector<float> host_correlated_rates_toaddCPU(dev_correlated_rates);      
      
    cudaThreadSynchronize();
    Timer h1;
 
    drift_add_device(adder_device,  // matrix to add
                                                                                                 dev_correlated_rates, // data to add to 
                                                                                                 n_steps, 
                                                                                                 number_rates ,
                                                                                                 n_vectors,
                                                                                                 useTexture); 

    cudaThreadSynchronize();
   	double time = h1.timePassed();
    std::cout << " time taken for all matrix additions."    << time  << std::endl;

    thrust::host_vector<float> host_correlated_rates_added(dev_correlated_rates);       
    
    Timer h2;
    
    for (int p = 0; p < n_vectors; ++p)
        for (int s=0; s < n_steps; ++s)
               for (int r=0; r < number_rates; ++r)     
                {                
                    int i = p +    n_vectors*(r+ s*number_rates);    
                    host_correlated_rates_toaddCPU[i] += adder[r+ s*number_rates];
               }
      
 
    double time2 = h2.timePassed();;
    std::cout << " time taken for all matrix additions."    << time2 << std::endl;               
    std::cout << "speed ratio: " << time2/time << "\n";                                
                                                                                                
    
    
      
      
                                                                                               
     double err1=0.0;
     double err2 = 0.0;
     
     for (int p = 0; p < n_vectors; ++p)
        for (int s=0; s < n_steps; ++s)
        {
                if (extraVerbose)
                    std::cout << p << "," << s << "\n";
      
            for (int r=0; r < number_rates; ++r)     
            {
                int i = p +    n_vectors*(r+ s*number_rates);
                
              
                
                if (extraVerbose)
                    std::cout <<  "," << host_correlated_rates_added[i]  << "," << host_correlated_rates_toaddCPU[i]    <<"\n";
                                        
                double erri =  host_correlated_rates_added[i]  -host_correlated_rates_toaddCPU[i] ;
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
