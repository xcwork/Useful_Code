//
//
//                                                                                                                                          LMM_evolver_full.cu
//
//
// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3

#include "LMM_evolver_full.h"
#include "LMM_evolver_all.h"
#include "LMM_evolver_main.h"
#include "LMM_evolver_gpu.h"

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
#include "correlate_drift_gpu.h"
#include "cudaMacros.h"
#include <gold/MatrixFacade.h> 

#include "Utilities.h"
#include "Brownian_bridge.h"
#include <gold/Bridge_allocator_gold.h>
// this routine does no memory allocation
// all memory should have already been allocated on the device
// using device vectors
// returns time taken

float LMMEvolutionRoutine(int paths, 
                                                       int pathOffset, 
                                                       int rates, 
                                                       int factors, 
                                                       int stepsForEvolution, 
                                                       int powerOfTwoForVariates,
                                                       thrust::device_vector<unsigned int>& scrambler_device, 
                                                       thrust::device_vector<float>& pseudoRoots_device,
                                                       thrust::device_vector<float>& fixedDrifts_device, 
                                                       thrust::device_vector<float>& displacements_device,
                                                       thrust::device_vector<float>& initial_rates_device, 
                                                       thrust::device_vector<float>& initial_log_rates_device, 
                                                       thrust::device_vector<float>& taus_device, 
                                                       thrust::device_vector<float>& initial_drifts_device, 
                                                       const std::vector<int>& aliveIndices, 
                                                        thrust::device_vector<int>& alive_device, 
// buffers
                                                       thrust::device_vector<unsigned int>& SobolInts_buffer_device, 
                                                       thrust::device_vector<float>& quasiRandoms_buffer_device, 
                                                       thrust::device_vector<float>& bridgeVariates_device, 
                                                       thrust::device_vector<float>& correlatedVariates_device, 
                                                       thrust::device_vector<float>& e_buffer_device,
                                                       thrust::device_vector<float>& e_buffer_pred_device,
                                                       thrust::device_vector<float>& evolved_rates_device, // for output
                                                       thrust::device_vector<float>& evolved_log_rates_device  // for output 
                                                        )
{
 
	Timer h1;

      bool useTextures = true;

      int tot_dimensions = intPower(2,powerOfTwoForVariates);
    
    // does some initializing and calls sobolGPUInts
    
      SobDeviceInts( paths, tot_dimensions*factors, pathOffset,SobolInts_buffer_device);
    
     cudaThreadSynchronize();

    // forwards to  scramble_gpu      
    // scrambling is in place
     scramble_main(SobolInts_buffer_device, // random numbers
                                                                                                scrambler_device,
                                                                                               tot_dimensions,
                                                                                                paths);
                                                                                                
    cudaThreadSynchronize();
                                                                            
    inverseCumulativeShawBrickmanUnsignedInt<float> inv;
       

    thrust::transform(SobolInts_buffer_device.begin(), SobolInts_buffer_device.end(),quasiRandoms_buffer_device.begin(),inv);
              
    cudaThreadSynchronize();
 
      
      
    BrownianBridgeMultiDim<float>::ordering allocator( BrownianBridgeMultiDim<float>::triangular);

// calls brownianBridgeMultiGPUReorder, and bridgeMainDevice
    MultiDBridge(paths, 
                                     powerOfTwoForVariates,
                                     factors,
                                     quasiRandoms_buffer_device, 
                                     bridgeVariates_device,
                                     allocator,
                                     useTextures)
                                    ;

      cudaThreadSynchronize();
    
 
// forwards to correlated_drift_paths_GPU
       correlated_drift_paths_device( bridgeVariates_device,
                                                                                correlatedVariates_device, // correlated rate increments 
                                                                                                 pseudoRoots_device, // correlator 
                                                                                                 alive_device,
                                                                                                 fixedDrifts_device,
                                                                                                  factors*rates,
                                                                                                  factors, 
                                                                                                  rates,
                                                                                                  paths,
                                                                                                  stepsForEvolution);
                                                                              
   // forwards raw pointers to LMM_evolver_pc_gpu                                                                   
    LMM_evolver_pc_main( initial_rates_device, 
                                                    initial_log_rates_device, 
                                                    taus_device, 
                                                    correlatedVariates_device,
                                                    pseudoRoots_device,
                                                    initial_drifts_device, 
                                                     displacements_device,
                                                    aliveIndices, 
                                                    paths,
                                                    factors,
                                                    stepsForEvolution, 
                                                    rates, 
                                                    e_buffer_device,
                                                    e_buffer_pred_device,
                                                    evolved_rates_device, // for output
                                                    evolved_log_rates_device  // for output 
                                                                                            );

    cudaThreadSynchronize();                               
      
   return static_cast<float>(h1.timePassed());
   
}

float LMMEvolutionRoutineRaw(int paths, 
                                                       int pathOffset, 
                                                       int rates, 
                                                       int factors, 
                                                       int stepsForEvolution, 
                                                       int powerOfTwoForVariates,
                                                       thrust::device_vector<unsigned int>& scrambler_device, 
                                                       thrust::device_vector<float>& pseudoRoots_device,
                                                       thrust::device_vector<float>& fixedDrifts_device, 
                                                       thrust::device_vector<float>& displacements_device,
                                                       thrust::device_vector<float>& initial_rates_device, 
                                                       thrust::device_vector<float>& initial_log_rates_device, 
                                                       thrust::device_vector<float>& taus_device, 
                                                       thrust::device_vector<float>& initial_drifts_device, 
                                                       const std::vector<int>& aliveIndices, 
                                                        thrust::device_vector<int>& alive_device, 
// buffers
                                                       thrust::device_vector<unsigned int>& SobolInts_buffer_device, 
                                                       thrust::device_vector<float>& quasiRandoms_buffer_device, 
                                                       thrust::device_vector<float>& bridgeVariates_device, 
                                                       float* correlatedVariates_global, 
                                                       float* e_buffer_global,
                                                       float* e_buffer_pred_global,
                                                       float* evolved_rates_global, // for output
                                                       float* evolved_log_rates_global  // for output 
                                                        )
{
    Timer h1;

      bool useTextures = true;

      int tot_dimensions = intPower(2,powerOfTwoForVariates);
    
    // does some initializing and calls sobolGPUInts
    
      SobDeviceInts( paths, tot_dimensions*factors, pathOffset,SobolInts_buffer_device);
    
     cudaThreadSynchronize();

    // forwards to  scramble_gpu      
    // scrambling is in place
     scramble_main(SobolInts_buffer_device, // random numbers
                                                                                                scrambler_device,
                                                                                               tot_dimensions,
                                                                                                paths);
                                                                                                
    cudaThreadSynchronize();
                                                                            
    inverseCumulativeShawBrickmanUnsignedInt<float> inv;
       

    thrust::transform(SobolInts_buffer_device.begin(), SobolInts_buffer_device.end(),quasiRandoms_buffer_device.begin(),inv);
              
    cudaThreadSynchronize();
 
      
      
    BrownianBridgeMultiDim<float>::ordering allocator( BrownianBridgeMultiDim<float>::triangular);

// calls brownianBridgeMultiGPUReorder, and bridgeMainDevice
    MultiDBridge(paths, 
                                     powerOfTwoForVariates,
                                     factors,
                                     quasiRandoms_buffer_device, 
                                     bridgeVariates_device,
                                     allocator,
                                     useTextures)
                                    ;

      cudaThreadSynchronize();
    
 
// forwards to correlated_drift_paths_GPU
       correlated_drift_paths_GPU( thrust::raw_pointer_cast(&bridgeVariates_device[0]),
                                                                                correlatedVariates_global, // correlated rate increments 
                                                                                                thrust::raw_pointer_cast(& pseudoRoots_device[0]), // correlator 
                                                                                                 thrust::raw_pointer_cast(&alive_device[0]),
                                                                                                 thrust::raw_pointer_cast(&fixedDrifts_device[0]),
                                                                                                  factors*rates,
                                                                                                  factors, 
                                                                                                  rates,
                                                                                                  paths,
                                                                                                  stepsForEvolution);
                                                                              
                                                                
    LMM_evolver_pc_gpu(thrust::raw_pointer_cast(& initial_rates_device[0]), 
        thrust::raw_pointer_cast(&initial_log_rates_device[0]), 
        thrust::raw_pointer_cast(&taus_device[0]), 
       correlatedVariates_global,
        thrust::raw_pointer_cast(&pseudoRoots_device[0]),
        thrust::raw_pointer_cast(&initial_drifts_device[0]), 
        thrust::raw_pointer_cast(&displacements_device[0]), 
        aliveIndices,
        paths,
        factors,
        stepsForEvolution, 
        rates,
        e_buffer_global, 
        e_buffer_pred_global, 
                                                    evolved_rates_global, // for output
                                                    evolved_log_rates_global  // for output 
                                                                                            );

    cudaThreadSynchronize();                               
    
   return static_cast<float>(h1.timePassed());
   
}


float LMMEvolutionRoutineRawSingleKernel(int paths, 
                                                       int pathOffset, 
                                                       int rates, 
                                                       int factors, 
                                                       int stepsForEvolution, 
                                                       int powerOfTwoForVariates,
                                                       thrust::device_vector<unsigned int>& scrambler_device, 
                                                       thrust::device_vector<float>& pseudoRoots_device,
                                                       thrust::device_vector<float>& fixedDrifts_device, 
                                                       thrust::device_vector<float>& displacements_device,
                                                       thrust::device_vector<float>& initial_rates_device, 
                                                       thrust::device_vector<float>& initial_log_rates_device, 
                                                       thrust::device_vector<float>& taus_device, 
                                                       thrust::device_vector<float>& initial_drifts_device, 
                                                       const std::vector<int>& aliveIndices, 
                                                        thrust::device_vector<int>& alive_device, 
// buffers
                                                       thrust::device_vector<unsigned int>& SobolInts_buffer_device, 
                                                       thrust::device_vector<float>& quasiRandoms_buffer_device, 
                                                       thrust::device_vector<float>& bridgeVariates_device, 
                                                       float* correlatedVariates_global, 
                                                       float* e_buffer_global,
                                                       float* e_buffer_pred_global,
                                                       float* evolved_rates_global,
                                                       float* evolved_log_rates_global,
                                                       float* discounts_global,
                                                       bool useSharedMem,
                                                       bool doDiscounts,
                                                       bool newBridge,
													   bool keplerArch,
													   int numberThreads
                                                        )
{

	Timer h1;

      bool useTextures = true;

      int tot_dimensions =stepsForEvolution;
		if (!newBridge)
			tot_dimensions= intPower(2,powerOfTwoForVariates);

	  if (static_cast<int>(scrambler_device.size()) < tot_dimensions)
		  scrambler_device.resize(tot_dimensions,0);

	  int sobolDimensionality = tot_dimensions*factors;
	  int variateDataSize = sobolDimensionality*paths;


	  SobolInts_buffer_device.resize(variateDataSize);
	  quasiRandoms_buffer_device.resize(variateDataSize);
    
    // does some initializing and calls sobolGPUInts
    
      SobDeviceInts( paths, tot_dimensions*factors, pathOffset,SobolInts_buffer_device);
    
     cudaThreadSynchronize();

    // forwards to  scramble_gpu      
    // scrambling is in place
     scramble_main(SobolInts_buffer_device, // random numbers
                                                                                                scrambler_device,
                                                                                               tot_dimensions,
                                                                                                paths);
                                                                                                
    cudaThreadSynchronize();
                                                                            
    inverseCumulativeShawBrickmanUnsignedInt<float> inv;
       

    thrust::transform(SobolInts_buffer_device.begin(), SobolInts_buffer_device.end(),quasiRandoms_buffer_device.begin(),inv);
              
    cudaThreadSynchronize();
 
    
    if (newBridge)
    {
        std::vector<int> indices(factors*stepsForEvolution);
        bridge_allocate_diagonal( indices,factors, stepsForEvolution);
        brownian_bridge bb(stepsForEvolution, indices);
        bb.transform(quasiRandoms_buffer_device, 
                   bridgeVariates_device,
                    paths, 
                    stepsForEvolution, 
                   factors);
    }

    else
    {
      
       BrownianBridgeMultiDim<float>::ordering allocator( BrownianBridgeMultiDim<float>::triangular);

// calls brownianBridgeMultiGPUReorder, and bridgeMainDevice
        MultiDBridge(paths, 
                                     powerOfTwoForVariates,
                                     factors,
                                     quasiRandoms_buffer_device, 
                                     bridgeVariates_device,
                                     allocator,
                                     useTextures)
                                    ;
    }
      cudaThreadSynchronize();
    
 
	if (keplerArch)
	 LMM_evolver_pc_all_gpu_kepler( thrust::raw_pointer_cast(& initial_rates_device[0]), 
                                                  thrust::raw_pointer_cast(&initial_log_rates_device[0]), 
                                                  thrust::raw_pointer_cast(&taus_device[0]), 
                                                  thrust::raw_pointer_cast(&bridgeVariates_device[0]),
                                                  thrust::raw_pointer_cast(&pseudoRoots_device[0]),
                                                  thrust::raw_pointer_cast(&initial_drifts_device[0]), 
                                                  thrust::raw_pointer_cast(&fixedDrifts_device[0]),
                                                  thrust::raw_pointer_cast(&displacements_device[0]), 
                                                  thrust::raw_pointer_cast(&alive_device[0]),
                                                  paths,
                                                  factors,
                                                  stepsForEvolution, 
                                                  rates, 
                                                  e_buffer_global,
                                                  e_buffer_pred_global,
                                                   evolved_rates_global, // for output     
                                                   evolved_log_rates_global,
                                                   discounts_global,
                                                   true, // bool transposedVariates,
												0//							  cudaStream_t streamNumber
                              );

	else
	    LMM_evolver_pc_all_gpu( thrust::raw_pointer_cast(& initial_rates_device[0]), 
                                                  thrust::raw_pointer_cast(&initial_log_rates_device[0]), 
                                                  thrust::raw_pointer_cast(&taus_device[0]), 
                                                  thrust::raw_pointer_cast(&bridgeVariates_device[0]),
                                                  thrust::raw_pointer_cast(&pseudoRoots_device[0]),
                                                  thrust::raw_pointer_cast(&initial_drifts_device[0]), 
                                                  thrust::raw_pointer_cast(&fixedDrifts_device[0]),
                                                  thrust::raw_pointer_cast(&displacements_device[0]), 
                                                  thrust::raw_pointer_cast(&alive_device[0]),
                                                  paths,
                                                  factors,
                                                  stepsForEvolution, 
                                                  rates, 
                                                  e_buffer_global,
                                                  e_buffer_pred_global,
                                                   evolved_rates_global, // for output     
                                                   evolved_log_rates_global,
                                                   discounts_global,
                                                   useSharedMem  ,                                                 
                                                   newBridge,
												   0,
												   numberThreads,
												   doDiscounts
                              );


    cudaThreadSynchronize();                               
      
    return static_cast<float>(h1.timePassed());
   
}


// this routine sets up the intermediate device vectors
float LMMEvolutionSemiDevRoutine(int paths, 
                                                       int pathOffset, 
                                                       int rates, 
                                                       int factors, 
                                                       int stepsForEvolution, 
                                                       int powerOfTwoForVariates,
                                                       const thrust::host_vector<unsigned int>& scrambler_host, 
                                                       int scramber_offset, 
                                                       const thrust::host_vector<float>& pseudoRoots_host,
                                                       const thrust::host_vector<float>& fixedDrifts_host, 
                                                        const thrust::host_vector<float>& displacements_host,
                                                       const thrust::host_vector<float>& initial_rates_host, 
                                                       const thrust::host_vector<float>& initial_log_rates_host, 
                                                       const thrust::host_vector<float>& taus_host, 
                                                       const thrust::host_vector<float>& initial_drifts_host, 
                                                       const std::vector<int>& aliveIndices, 
                                                       const thrust::host_vector<int>& alive_host, 
                                                       thrust::device_vector<float>& evolved_rates_device, // for output
                                                      thrust::device_vector<float>& evolved_log_rates_device // for output
                                                        )
{
  int tot_dimensions = intPower(2,powerOfTwoForVariates);
   int numberTotalVariates= paths*factors*tot_dimensions;
   int numberEvolvedRates = paths*rates*stepsForEvolution;
 thrust::device_vector<unsigned int> SobolInts_buffer_device(numberTotalVariates);
   
   thrust::device_vector<float> quasiRandoms_buffer_device(numberTotalVariates);                                                      
   thrust::device_vector<float> bridgeVariates_device(numberTotalVariates);
   thrust::device_vector<float> correlatedVariates_device(numberEvolvedRates);
   thrust::device_vector<float> e_buffer_device(paths*factors);
   thrust::device_vector<float> e_buffer_pred_device(paths*factors);


   Timer h1;
    
   evolved_rates_device.resize(numberEvolvedRates);
   evolved_log_rates_device.resize(numberEvolvedRates);

   // now pass input data to GPU
   
   
   thrust::device_vector<unsigned int> scrambler_device(factors*tot_dimensions); 
   thrust::copy(scrambler_host.begin()+scramber_offset,scrambler_host.begin()+scramber_offset+scrambler_device.size(),scrambler_device.begin() );
   
   
   thrust::device_vector<float> pseudoRoots_device(pseudoRoots_host);   
   thrust::device_vector<float> fixedDrifts_device(fixedDrifts_host); 
   thrust::device_vector<float> displacements_device(displacements_host);
   thrust::device_vector<float> initial_rates_device(initial_rates_host); 
   thrust::device_vector<float> initial_log_rates_device(initial_log_rates_host); 
   thrust::device_vector<float> taus_device(taus_host);
   thrust::device_vector<float> initial_drifts_device(initial_drifts_host);
   thrust::device_vector<int> alive_device(alive_host);
   
  float   innerTime =  LMMEvolutionRoutine(paths, 
                                                       pathOffset, 
                                                       rates, 
                                                       factors, 
                                                       stepsForEvolution, 
                                                       powerOfTwoForVariates,
                                                       scrambler_device, 
                                                       pseudoRoots_device,
                                                       fixedDrifts_device, 
                                                       displacements_device,
                                                       initial_rates_device, 
                                                       initial_log_rates_device, 
                                                       taus_device, 
                                                       initial_drifts_device, 
                                                       aliveIndices, 
                                                       alive_device, 
                                                       SobolInts_buffer_device, 
                                                       quasiRandoms_buffer_device, 
                                                       bridgeVariates_device, 
                                                       correlatedVariates_device, 
                                                       e_buffer_device,
                                                       e_buffer_pred_device,
                                                       evolved_rates_device, // for output
                                                       evolved_log_rates_device  // for output 
                                                        );
     cudaThreadSynchronize();                               
      
                                                          
                                                        
    std::cout << " inner time " << innerTime << "\n";   
    std::cout << " semi dev time " << h1.timePassed() << "\n";                                                            
                                                        
   return innerTime;
                                                           
}                                                        

// this routine sets up the device vectors
float LMMEvolutionMainRoutine(int paths, 
                                                       int pathOffset, 
                                                       int rates, 
                                                       int factors, 
                                                       int stepsForEvolution, 
                                                       int powerOfTwoForVariates,
                                                       const thrust::host_vector<unsigned int>& scrambler_host, 
                                                       const thrust::host_vector<float>& pseudoRoots_host,
                                                       const thrust::host_vector<float>& fixedDrifts_host, 
                                                        const thrust::host_vector<float>& displacements_host,
                                                       const thrust::host_vector<float>& initial_rates_host, 
                                                       const thrust::host_vector<float>& initial_log_rates_host, 
                                                       const thrust::host_vector<float>& taus_host, 
                                                       const thrust::host_vector<float>& initial_drifts_host, 
                                                       const std::vector<int>& aliveIndices, 
                                                       const thrust::host_vector<int>& alive_host, 
                                                       thrust::host_vector<float>& evolved_rates_host, // for output
                                                       bool getLogs, 
                                                       thrust::host_vector<float>& evolved_log_rates_host,// for output
                                                       float& innerTime  // for output 
                                                        )
{

   Timer h1;

   int tot_dimensions = intPower(2,powerOfTwoForVariates);
//   int numberTotalVariates= paths*factors*tot_dimensions;
  // int numberUsefulVariates =  paths*factors*stepsForEvolution;
   int numberEvolvedRates = paths*rates*stepsForEvolution;
   
   
   cudaSetDevice(cutGetMaxGflopsDeviceId());
  {
  
  // allocate empty memory on GPU
 
   
   thrust::device_vector<float> evolved_rates_device(numberEvolvedRates);                                                       
   thrust::device_vector<float> evolved_log_rates_device(numberEvolvedRates);
   
   int scrambleOffset =0;
   
     innerTime=LMMEvolutionSemiDevRoutine(paths, 
                                                       pathOffset, 
                                                       rates, 
                                                       factors, 
                                                       stepsForEvolution, 
                                                       powerOfTwoForVariates,
                                                       scrambler_host, 
                                                       scrambleOffset,
                                                       pseudoRoots_host,
                                                       fixedDrifts_host, 
                                                       displacements_host,
                                                       initial_rates_host, 
                                                       initial_log_rates_host, 
                                                       taus_host, 
                                                       initial_drifts_host, 
                                                       aliveIndices, 
                                                       alive_host, 
                                                       evolved_rates_device, // for output
                                                       evolved_log_rates_device  // for output 
                                                        );
                                                           
 
                                                        
    evolved_rates_host=evolved_rates_device;
    
    
    if (getLogs)
        evolved_log_rates_host = evolved_log_rates_device;
    }
                                               
    cudaThreadExit();
   
   return static_cast<float>(h1.timePassed());
}
