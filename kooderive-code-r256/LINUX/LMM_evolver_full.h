
//
//
//                                                                                                                                          LMM_evolver_full.h
//
//
// (c) Mark Joshi 2010
// This code is released under the GN

#ifndef LMM_EVOLVER_FULL_H
#define LMM_EVOLVER_FULL_H
#include <gold/pragmas.h> 
#include <thrust/device_vector.h>


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
                                                        );


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
                                                        );



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
													   int threadsForLMMKernel
                                                        );

// this routine sets up the intermediate device vectors
// and then calls the device vector routine
// this allows the possibility of only keeping the useful data allocated 
float LMMEvolutionSemiDevRoutine(int paths, 
                                                       int pathOffset, 
                                                       int rates, 
                                                       int factors, 
                                                       int stepsForEvolution, 
                                                       int powerOfTwoForVariates,
                                                       const thrust::host_vector<unsigned int>& scrambler_host, 
                                                       int scramble_offset,
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
                                                        );

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
                                                       thrust::host_vector<float>& evolved_log_rates_host,  // for output 
                                                       float& innerTime
                                                        );

#endif
