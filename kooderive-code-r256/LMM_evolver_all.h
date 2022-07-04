/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
//
//
//                                                LMM_evolver_all_gpu.h
//
//
//

// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3

/*
    do all the drifts and rate increments in one kernel
*/


#ifndef LMM_EVOLVER_ALL
#define LMM_EVOLVER_ALL
#include <cutil_inline.h>


extern "C"
void LMM_evolver_pc_all_gpu(  float* initial_rates_device, 
                              float* initial_log_rates_device, 
                              float* taus_device, 
                              float* variates_device,
                              float* pseudo_roots_device,
                              float* initial_drifts_device, 
                              float* fixed_drifts_device, 
                              float* displacements_device,
                              int* alive_device,
                              int paths,
                              int factors,
                              int steps, 
                              int rates, 
                              float* e_buffer_device,
                              float* e_buffer_pred_device,
                              float* evolved_rates_device, // for output
                              float* evolved_log_rates_device,
                              float* discounts_device,
                              bool sharedMemForLogs,
                              bool transposeData,
							  cudaStream_t streamNumber,
							  int numberThreads,
							   bool doDiscounts 
                              );

void LMM_evolver_pc_all_gpu_kepler(  float* initial_rates_global, 
                              float* initial_log_rates_global, 
                              float* taus_global, 
                              float* variates_global,
                              float* pseudo_roots_global,
                              float* initial_drifts_global, 
                              float* fixed_drifts_global, 
                              float* displacements_global,
                              int* alive_global,
                              int paths,
                              int factors,
                              int steps, 
                              int rates, 
                              float* e_buffer_global,
                              float* e_buffer_pred_global,
                              float* evolved_rates_global, // for output
                              float* evolved_log_rates_global,
                              float* discounts_global,
                              bool transposedVariates,
							    int threadsperblock,
							  cudaStream_t streamNumber=0
							
                              );


extern "C"
void LMM_evolver_pc_all_fermi_gpu(  float* initial_rates_global, 
                              float* initial_log_rates_global, 
                              float* taus_global, 
                              float* variates_global,
                              float* pseudo_roots_global,
                              float* initial_drifts_global, 
                              float* fixed_drifts_global, 
                              float* displacements_global,
                              int* alive_global,
                              int paths,
                              int factors,
                              int steps, 
                              int rates, 
                              float* e_buffer_global,
                              float* e_buffer_pred_global,
                              float* evolved_rates_global, // for output
                              float* evolved_log_rates_global,
                              float* discounts_global,
                              bool transposedVariates,
							  cudaStream_t streamNumber
                              );

#endif
