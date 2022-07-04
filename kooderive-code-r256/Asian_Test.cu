//
//
//                                                                                                                                          Asian_Test.cu
//
//
// (c) Mark Joshi 2009,2010
// This code is released under the GNU public licence version 3

#include "Asian_Test.h"
#include <gold/Asian_gold.h>
#include "Asian.h"

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
#include <gold/Bridge_gold.h>
#include "sobol.h"

#include "Bridge.h"
#include "cudaMacros.h"

#include "Utilities.h"
#include "DeviceChooser.h"
// compare GPU against CPU


namespace
{
    double tolerance = 1E-5;
}
int AsianTestRoutine(bool verbose, DeviceChooser & chooser)
{
double L1arith, L1Geom;

{
    unsigned path_offset=0;

    int n_vectors =  intPower(2,21);
    int n_poweroftwo =8;
    int n_dimensions = intPower(2,n_poweroftwo);
 
    size_t N= n_vectors*n_dimensions; 




   cudaSetDevice(chooser.WhichDevice());

   size_t estimatedFloatsRequired = 4*(2*N+2*n_vectors);
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
               N= n_vectors*n_dimensions; 
  //             change =true;
               estimatedFloatsRequired = 4*(2*N+2*n_vectors);
               estimatedMem = sizeof(float)*estimatedFloatsRequired;
               std::cout << " checkinh ";

           }
       }

       
    cudaThreadSynchronize();
       
     thrust::device_vector<float> dev_output(N);    
     thrust::device_vector<float> device_input(N);
     
     thrust::device_vector<float> outputDataArithmetic(n_vectors);   
     thrust::device_vector<float> outputDataGeometric(n_vectors);
     
     cutilSafeCall(cudaThreadSynchronize());
   
      cudaThreadSynchronize();
                  
      bool doNormalInSobol=true;
      SobDevice( n_vectors, n_dimensions, path_offset, device_input,doNormalInSobol);
     

      cudaThreadSynchronize();
        
       
        
     bridgeMainDevice(n_vectors, 
                                                n_poweroftwo,
                                                device_input,
                                                dev_output);

            	 
       cudaThreadSynchronize();


   // option data
   float spot =100.0f;
   float expiry =1.0f;
   float strike = 100.0; // 80.0f+20.0f*(j % 3);
   float r = 0.05f;
   float d = 0.03f;
   float sigma = 0.1f +0.002f;
   int n_steps = n_dimensions;
   
   float logSpot = log(spot);
   float df = exp(-r*expiry);
   
   int samplingDates =n_dimensions;
   
   if (samplingDates > n_steps)
   {    
       samplingDates = n_steps;
       std::cout << "sampling dates must be less than or equal to n_steps and so has been reduced to "<< samplingDates <<"\n";
   }
   
  std::vector<float> logDrifts_vec(samplingDates);
  std::vector<float> logSds_vec(samplingDates);
  
  float dt =  expiry/samplingDates;
  
  for (int k=0; k < samplingDates; ++k)
  {          
        logSds_vec[k] = sigma*sqrt(dt);
           
           
        logDrifts_vec[k] = (r-d)*dt-0.5f*logSds_vec[k]*logSds_vec[k];

  }
   
   float strikeArithmetic = strike;
   float strikeGeometric = strike;
   
   
      std::cout << " \n\nenter  Asian\n";    
   
   AsianMainDevice(
                        dev_output,
                           n_dimensions, // must be greater than or equal to stepsPerPath
                                                                                                n_vectors, 
                                                                                                samplingDates,
                                                                                                 logDrifts_vec, 
                                                                                                 logSds_vec, 
                                                                                                 logSpot,
                                                                                                 df,
                                                                                                 strikeArithmetic,
                                                                                                 outputDataArithmetic,
                                                                                                 strikeGeometric,
                                                                                                 outputDataGeometric );
                                                                                                 
    thrust::host_vector<float>     outputDataArithmeticFromGPU(   outputDataArithmetic.size());
    outputDataArithmeticFromGPU = outputDataArithmetic;
     
    thrust::host_vector<float>    outputDataGeometricFromGPU(   outputDataGeometric.size());
    outputDataGeometricFromGPU = outputDataGeometric;
                                                                                                 
    std::vector<float> outputDataArithmeticVec, outputDataGeometricVec;
    
    thrust::host_vector<float> host_out(dev_output);
    
    AsianCallGPU_gold (
                        &host_out[0],
                           n_dimensions, 
                                                                                                n_vectors, 
                                                                                                samplingDates,
                                                                                                 logDrifts_vec, 
                                                                                                 logSds_vec, 
                                                                                                 logSpot,
                                                                                                 df,
                                                                                                 strikeArithmetic,
                                                                                                 outputDataArithmeticVec,
                                                                                                 strikeGeometric,
                                                                                                 outputDataGeometricVec );

                                                                                                
     double errArith=0.0;
     double errGeom =0.0;
     
     for (int i=0; i < n_vectors; ++i)
     {
        errArith +=    fabs( outputDataArithmeticFromGPU[i] -  outputDataArithmeticVec[i]);           
        errGeom +=    fabs( outputDataGeometricFromGPU[i] -  outputDataGeometricVec[i]); 
         
    }
           
    L1arith = errArith/n_vectors;
    L1Geom = errGeom/n_vectors;
      
    if (verbose)
    {
        std::cout << "L1 arithmetic error " << L1arith << "L1 Geometric error " << L1Geom << "\n";
    }
    
    }
    
        
    cudaThreadExit();
      
    if (L1arith > tolerance)
    {
        std::cout << " Asian option test failed";
        return 1;
    }
    
    if (L1Geom > tolerance)
    {
        std::cout << " Asian option geometric test failed";
        return 1;
     }   

     std::cout << " Asian option  test passed";
      
 
    return 0;
}


