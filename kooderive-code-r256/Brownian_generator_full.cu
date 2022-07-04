//
//
//                                                                                                                                          LMM_evolver_full.cu
//
//
// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3

#include "Brownian_generator_full.h"


#include "multid_path_gen_BS_main.h"
#include "scramble_main.h"

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
#include "correlate_drift_main.h"
#include "cudaMacros.h"
#include <gold/MatrixFacade.h> 

#include "Utilities.h"
#include <gpuCompatibilityCheck.h>
// this routine does no memory allocation
// all memory should have already been allocated on the device
// using device vectors
// returns time taken

float BrownianGenerationRoutine(int paths, 
                                                        int pathOffset, 
                                                        int factors, 
                                                        int stepsForEvolution, 
                                                        int powerOfTwoForVariates,
                                                        thrust::device_vector<unsigned int>& scrambler_device, 
// buffers
                                                        thrust::device_vector<unsigned int>& SobolInts_buffer_device, 
                                                        thrust::device_vector<float>& quasiRandoms_buffer_device, 
                                                        thrust::device_vector<float>& bridgeVariates_device
                                                         )
{
  Timer h1;

      bool useTextures = true;

      int tot_dimensions = intPower(2,powerOfTwoForVariates);
    
      SobDeviceInts( paths, tot_dimensions*factors, pathOffset,SobolInts_buffer_device);
      


     cudaThreadSynchronize();
        
     // scrambling is in place
     scramble_main(SobolInts_buffer_device, // random numbers
                                                                                                scrambler_device,
                                                                                               tot_dimensions,
                                                                                                paths);
                                                                                                
    cudaThreadSynchronize();
                                                                              
    inverseCumulativeShawBrickmanUnsignedInt<float> inv;
       

    thrust::transform(SobolInts_buffer_device.begin(), SobolInts_buffer_device.end(),quasiRandoms_buffer_device.begin(),inv);
              
    cudaThreadSynchronize();
/*
#ifdef _DEBUG

{
       
        thrust::host_vector<float> quasiRandoms_buffer_host(quasiRandoms_buffer_device);
        std::vector<float> quasiRandoms_buffer_vec(quasiRandoms_buffer_host.size());
        std::copy(quasiRandoms_buffer_host.begin(),quasiRandoms_buffer_host.end(),quasiRandoms_buffer_vec.begin());
        
       std::cout << "  quasiRandoms_buffer_host in BrownianGenerationRoutine \n";
       
       for (int i=0; i < quasiRandoms_buffer_vec.size(); ++i)
            std::cout << quasiRandoms_buffer_vec[i] << "\n";
        
        
}

#endif
*/

      
      
    BrownianBridgeMultiDim<float>::ordering allocator( BrownianBridgeMultiDim<float>::triangular);


    MultiDBridge(paths, 
                                     powerOfTwoForVariates,
                                     factors,
                                     quasiRandoms_buffer_device, 
                                     bridgeVariates_device,
                                     allocator,
                                     useTextures)
                                    ;

      cudaThreadSynchronize();
      /*
#ifdef _DEBUG

{
        thrust::host_vector<float> bridgeVariates_host(bridgeVariates_device);
        std::vector<float> bridgeVariates_vec(bridgeVariates_host.size());
        std::copy(bridgeVariates_host.begin(),bridgeVariates_host.end(),bridgeVariates_vec.begin());
        
    CubeConstFacade<float> variatesFacade(&bridgeVariates_vec[0],stepsForEvolution,paths,factors);
    
    std::cout << "  \n\nbridgeVariates_device in BrownianGenerationRoutine \n";
 
    for (int p=0; p < paths; ++p)
        for (int s=0; s < stepsForEvolution; ++s)
        {       
            std::cout << "s ," << s << ", p ," << p <<"," ;
      
            for (int f=0; f < factors; ++f)    
                 std::cout << variatesFacade(s,p,f)  << ",";
        
            std::cout << "\n";
        }
      

        

}

#endif
*/
          
  
   return static_cast<float>(h1.timePassed());
   
}


// this routine sets up the device vectors
float BrownianGenerationMainRoutine(int paths, 
                                                       int pathOffset, 
                                                       int factors, 
                                                       int stepsForEvolution, 
                                                       int powerOfTwoForVariates,
                                                       const thrust::host_vector<unsigned int>& scrambler_host, 
                                                       thrust::host_vector<float>& bridgeVariates_host,
                                                       float &innerTime
                                                        )
{

   Timer h1;

   int tot_dimensions = intPower(2,powerOfTwoForVariates);
   int numberTotalVariates= paths*factors*tot_dimensions;
   
   
   cudaSetDevice(cutGetMaxGflopsDeviceId());
  {
  
  // allocate empty memory on GPU
   thrust::device_vector<unsigned int> SobolInts_buffer_device(numberTotalVariates);
   
   thrust::device_vector<float> quasiRandoms_buffer_device(numberTotalVariates);                                                      
   thrust::device_vector<float> bridgeVariates_device(numberTotalVariates);
     
   // now pass input data to GPU
   thrust::device_vector<unsigned int> scrambler_device(scrambler_host); 
     
   innerTime =  BrownianGenerationRoutine(paths, 
                                                       pathOffset, 
                                                       factors, 
                                                       stepsForEvolution, 
                                                       powerOfTwoForVariates,
                                                       scrambler_device, 
                                                       SobolInts_buffer_device, 
                                                       quasiRandoms_buffer_device, 
                                                       bridgeVariates_device
                                                        );
                                                        
    bridgeVariates_host =bridgeVariates_device;

    }
                                               
    cudaThreadExit();
     
   return static_cast<float>(h1.timePassed());
}
