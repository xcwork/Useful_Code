
//
//
//                                        LMM Evolver main .cu
//
//
// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3

#include "LMM_evolver_main.h"
#include <iostream>
#include <cutil_inline.h>
#include <math.h>
#include <thrust/version.h>
#include <vector>
#include <thrust/device_vector.h>
#include <gold/Errors.h>
#include "LMM_evolver_gpu.h"
#include "LMM_evolver_all.h"

void LMM_evolver_euler_main(  thrust::device_vector<float>& initial_rates_device, 
                            thrust::device_vector<float>& initial_log_rates_device, 
                            thrust::device_vector<float>& taus_device, 
                            thrust::device_vector<float>& correlatedBrownianIncrements_device,
                            thrust::device_vector<float>& pseudo_roots_device,
                            thrust::device_vector<float>& initial_drifts_device, 
                            thrust::device_vector<float>& displacements_device, 
                            const std::vector<int>& aliveIndices,
                            int paths,
                            int factors,
                            int steps, 
                            int rates, 
                            thrust::device_vector<float>&  e_buffer_device,
                            thrust::device_vector<float>& evolved_rates_device, // for output
                            thrust::device_vector<float>&  evolved_log_rates_device  // for output 
                            )
{
    LMM_evolver_euler_gpu(  thrust::raw_pointer_cast(& initial_rates_device[0]), 
        thrust::raw_pointer_cast(&initial_log_rates_device[0]), 
        thrust::raw_pointer_cast(&taus_device[0]), 
        thrust::raw_pointer_cast(& correlatedBrownianIncrements_device[0]),
        thrust::raw_pointer_cast(&pseudo_roots_device[0]),
        thrust::raw_pointer_cast(&initial_drifts_device[0]), 
        thrust::raw_pointer_cast(&displacements_device[0]), 
        aliveIndices,
        paths,
        factors,
        steps, 
        rates,
        thrust::raw_pointer_cast(&e_buffer_device[0]), 
        thrust::raw_pointer_cast(&evolved_rates_device[0]), // for output
        thrust::raw_pointer_cast(&evolved_log_rates_device[0])  // for output 
        );
}


void LMM_evolver_pc_main(  thrust::device_vector<float>& initial_rates_device, 
                         thrust::device_vector<float>& initial_log_rates_device, 
                         thrust::device_vector<float>& taus_device, 
                         thrust::device_vector<float>& correlatedBrownianIncrements_device,
                         thrust::device_vector<float>& pseudo_roots_device,
                         thrust::device_vector<float>& initial_drifts_device, 
                         thrust::device_vector<float>& displacements_device, 
                         const std::vector<int>& aliveIndices,
                         int paths,
                         int factors,
                         int steps, 
                         int rates, 
                         thrust::device_vector<float>&  e_buffer_device,
                         thrust::device_vector<float>&  e_buffer_pred_device,
                         thrust::device_vector<float>& evolved_rates_device, // for output
                         thrust::device_vector<float>&  evolved_log_rates_device  // for output 
                         )
{
    LMM_evolver_pc_gpu(  thrust::raw_pointer_cast(& initial_rates_device[0]), 
        thrust::raw_pointer_cast(&initial_log_rates_device[0]), 
        thrust::raw_pointer_cast(&taus_device[0]), 
        thrust::raw_pointer_cast(& correlatedBrownianIncrements_device[0]),
        thrust::raw_pointer_cast(&pseudo_roots_device[0]),
        thrust::raw_pointer_cast(&initial_drifts_device[0]), 
        thrust::raw_pointer_cast(&displacements_device[0]), 
        aliveIndices,
        paths,
        factors,
        steps, 
        rates,
        thrust::raw_pointer_cast(&e_buffer_device[0]), 
        thrust::raw_pointer_cast(&e_buffer_pred_device[0]), 
        thrust::raw_pointer_cast(&evolved_rates_device[0]), // for output
        thrust::raw_pointer_cast(&evolved_log_rates_device[0])  // for output 
        );
}

void LMM_evolver_pc_single_kernel_main(  thrust::device_vector<float>& initial_rates_device, 
                         thrust::device_vector<float>& initial_log_rates_device, 
                         thrust::device_vector<float>& taus_device, 
                         thrust::device_vector<float>& variates_device,
                         thrust::device_vector<float>& pseudo_roots_device,
                         thrust::device_vector<float>& initial_drifts_device, 
                         thrust::device_vector<float>& fixed_drifts_device, 
                         thrust::device_vector<float>& displacements_device, 
                           thrust::device_vector<int>& alive_device,
                         int paths,
                         int factors,
                         int steps, 
                         int rates, 
                         thrust::device_vector<float>&  e_buffer_device,
                         thrust::device_vector<float>&  e_buffer_pred_device,
                         thrust::device_vector<float>& evolved_rates_device,
                         thrust::device_vector<float>& evolved_log_rates_device,
                        bool useSharedMemForLogs
                         )
{
 LMM_evolver_pc_all_gpu(  thrust::raw_pointer_cast(& initial_rates_device[0]), 
        thrust::raw_pointer_cast(&initial_log_rates_device[0]), 
        thrust::raw_pointer_cast(&taus_device[0]), 
        thrust::raw_pointer_cast(& variates_device[0]),
        thrust::raw_pointer_cast(&pseudo_roots_device[0]),
        thrust::raw_pointer_cast(&initial_drifts_device[0]), 
        thrust::raw_pointer_cast(&fixed_drifts_device[0]), 
        thrust::raw_pointer_cast(&displacements_device[0]), 
        thrust::raw_pointer_cast(&alive_device[0]), 
        paths,
        factors,
        steps, 
        rates,
        thrust::raw_pointer_cast(&e_buffer_device[0]), 
        thrust::raw_pointer_cast(&e_buffer_pred_device[0]), 
        thrust::raw_pointer_cast(&evolved_rates_device[0]),
        thrust::raw_pointer_cast(&evolved_log_rates_device[0]),
        0, // null pointer since discounts device not used
        useSharedMemForLogs, 
         false, // no transposed data
        0,
		0,
		false 
		 );
}



void LMM_evolver_pc_single_kernel_discounts_main(  thrust::device_vector<float>& initial_rates_device, 
                         thrust::device_vector<float>& initial_log_rates_device, 
                         thrust::device_vector<float>& taus_device, 
                         thrust::device_vector<float>& variates_device,
                         thrust::device_vector<float>& pseudo_roots_device,
                         thrust::device_vector<float>& initial_drifts_device, 
                         thrust::device_vector<float>& fixed_drifts_device, 
                         thrust::device_vector<float>& displacements_device, 
                           thrust::device_vector<int>& alive_device,
                         int paths,
                         int factors,
                         int steps, 
                         int rates, 
                         thrust::device_vector<float>&  e_buffer_device,
                         thrust::device_vector<float>&  e_buffer_pred_device,
                         thrust::device_vector<float>& evolved_rates_device,
                         thrust::device_vector<float>& evolved_log_rates_device,
                         thrust::device_vector<float>& discounts_device,
                        bool useSharedMemForLogs,
                        bool doDiscounts,
                        bool transposeData,
						int threads
                         )
{
 LMM_evolver_pc_all_gpu(  thrust::raw_pointer_cast(& initial_rates_device[0]), 
        thrust::raw_pointer_cast(&initial_log_rates_device[0]), 
        thrust::raw_pointer_cast(&taus_device[0]), 
        thrust::raw_pointer_cast(& variates_device[0]),
        thrust::raw_pointer_cast(&pseudo_roots_device[0]),
        thrust::raw_pointer_cast(&initial_drifts_device[0]), 
        thrust::raw_pointer_cast(&fixed_drifts_device[0]), 
        thrust::raw_pointer_cast(&displacements_device[0]), 
        thrust::raw_pointer_cast(&alive_device[0]), 
        paths,
        factors,
        steps, 
        rates,
        thrust::raw_pointer_cast(&e_buffer_device[0]), 
        thrust::raw_pointer_cast(&e_buffer_pred_device[0]), 
        thrust::raw_pointer_cast(&evolved_rates_device[0]),
        thrust::raw_pointer_cast(&evolved_log_rates_device[0]),
        thrust::raw_pointer_cast(&discounts_device[0]), 
        useSharedMemForLogs, 
        transposeData,
		0,threads, // dummies for stream and threads
		doDiscounts
        );
}

void LMM_evolver_pc_single_kernel_discounts_kepler_main(  thrust::device_vector<float>& initial_rates_device, 
                         thrust::device_vector<float>& initial_log_rates_device, 
                         thrust::device_vector<float>& taus_device, 
                         thrust::device_vector<float>& variates_device,
                         thrust::device_vector<float>& pseudo_roots_device,
                         thrust::device_vector<float>& initial_drifts_device, 
                         thrust::device_vector<float>& fixed_drifts_device, 
                         thrust::device_vector<float>& displacements_device, 
                           thrust::device_vector<int>& alive_device,
                         int paths,
                         int factors,
                         int steps, 
                         int rates, 
                         thrust::device_vector<float>&  e_buffer_device,
                         thrust::device_vector<float>&  e_buffer_pred_device,
                         thrust::device_vector<float>& evolved_rates_device,
                         thrust::device_vector<float>& evolved_log_rates_device,
                         thrust::device_vector<float>& discounts_device,
                        bool transposeData,
						int threadsperblock
                         )
{
 LMM_evolver_pc_all_gpu_kepler(  thrust::raw_pointer_cast(& initial_rates_device[0]), 
        thrust::raw_pointer_cast(&initial_log_rates_device[0]), 
        thrust::raw_pointer_cast(&taus_device[0]), 
        thrust::raw_pointer_cast(& variates_device[0]),
        thrust::raw_pointer_cast(&pseudo_roots_device[0]),
        thrust::raw_pointer_cast(&initial_drifts_device[0]), 
        thrust::raw_pointer_cast(&fixed_drifts_device[0]), 
        thrust::raw_pointer_cast(&displacements_device[0]), 
        thrust::raw_pointer_cast(&alive_device[0]), 
        paths,
        factors,
        steps, 
        rates,
        thrust::raw_pointer_cast(&e_buffer_device[0]), 
        thrust::raw_pointer_cast(&e_buffer_pred_device[0]), 
        thrust::raw_pointer_cast(&evolved_rates_device[0]),
        thrust::raw_pointer_cast(&evolved_log_rates_device[0]),
        thrust::raw_pointer_cast(&discounts_device[0]), 
        transposeData,
		threadsperblock
        );
}


void LMM_evolver_pc_all_fermi_main(  thrust::device_vector<float>& initial_rates_device, 
                              thrust::device_vector<float>& initial_log_rates_device, 
                              thrust::device_vector<float>& taus_device, 
                              thrust::device_vector<float>& variates_device,
                              thrust::device_vector<float>& pseudo_roots_device,
                              thrust::device_vector<float>& initial_drifts_device, 
                              thrust::device_vector<float>& fixed_drifts_device, 
                              thrust::device_vector<float>& displacements_device,
                              thrust::device_vector<int>& alive_device,
                              int paths,
                              int factors,
                              int steps, 
                              int rates, 
                              thrust::device_vector<float>& e_buffer_device,
                              thrust::device_vector<float>& e_buffer_pred_device,
                              thrust::device_vector<float>& evolved_rates_device, // for output
                              thrust::device_vector<float>& evolved_log_rates_device,
                              thrust::device_vector<float>& discounts_device,
                              bool transposedVariates,
							  cudaStream_t streamNumber
                              )
{
    LMM_evolver_pc_all_fermi_gpu(   thrust::raw_pointer_cast(& initial_rates_device[0]), 
                              thrust::raw_pointer_cast(&  initial_log_rates_device[0]), 
                               thrust::raw_pointer_cast(& taus_device[0]), 
                               thrust::raw_pointer_cast(&  variates_device[0]),
                               thrust::raw_pointer_cast(&  pseudo_roots_device[0]),
                               thrust::raw_pointer_cast(&  initial_drifts_device[0]), 
                               thrust::raw_pointer_cast(&  fixed_drifts_device[0]), 
                               thrust::raw_pointer_cast(&  displacements_device[0]),
                               thrust::raw_pointer_cast(&  alive_device[0]),
                              paths,
                               factors,
                              steps, 
                              rates, 
                               thrust::raw_pointer_cast(& e_buffer_device[0]),
                               thrust::raw_pointer_cast(& e_buffer_pred_device[0]),
                               thrust::raw_pointer_cast(&  evolved_rates_device[0]), // for output
                               thrust::raw_pointer_cast(& evolved_log_rates_device[0]),
                               thrust::raw_pointer_cast(&  discounts_device[0]),
                              transposedVariates,
							  streamNumber
                              );

}

void discount_ratios_computation_main( thrust::device_vector<float>& evolved_rates_device, 
                                      thrust::device_vector<float>& taus_device, 
                                      const std::vector<int>& aliveIndices, 
                                      thrust::device_vector<int>& alive_device, 
                                      int paths,
                                      int steps, 
                                      int rates, 
                                      thrust::device_vector<float>& discounts_device,  // for output 
                                      bool allStepsAtOnce
                                      )
{

    discount_ratios_computation_gpu(    thrust::raw_pointer_cast(& evolved_rates_device[0]), 
        thrust::raw_pointer_cast(& taus_device[0]), 
        aliveIndices, 
        thrust::raw_pointer_cast(& alive_device[0]), 
        paths,
        steps, 
        rates, 
        thrust::raw_pointer_cast(&discounts_device[0]) ,
        allStepsAtOnce
        );

}


void coterminal_annuity_ratios_computation_gpu(  thrust::device_vector<float>& discounts_device, 
                                               thrust::device_vector<float>&taus_device, 
                                               const std::vector<int>& aliveIndices, 
                                               int paths,
                                               int steps, 
                                               int rates, 
                                               thrust::device_vector<float>& annuities_device  // for output 
                                               )

{                                                                                            
    coterminal_annuity_ratios_computation_gpu(   thrust::raw_pointer_cast(&discounts_device[0])  , 
        thrust::raw_pointer_cast(& taus_device[0])  , 
        aliveIndices, 
        paths,
        steps, 
        rates, 
        thrust::raw_pointer_cast(& annuities_device[0])    // for output 
        );
}



void coterminal_swap_rates_computation_main_gpu(  thrust::device_vector<float>& discounts_device, 
                                           thrust::device_vector<float>& annuities_device,
                                           const std::vector<int>& aliveIndices, 
                                           int paths,
                                           int steps, 
                                           int rates, 
                                           thrust::device_vector<float>& cot_swap_rates_device  // for output 
                                           )
{
    coterminal_swap_rate_computation_gpu(  thrust::raw_pointer_cast(&discounts_device[0])  , 
        thrust::raw_pointer_cast(& annuities_device[0]),
        aliveIndices, 
        paths,
        steps, 
        rates, 
        thrust::raw_pointer_cast(&cot_swap_rates_device[0]) 
        );

}

void swap_rate_computation_gpu( thrust::device_vector<float>& discounts_device, 
                               thrust::device_vector<float>&taus_device, 
                               int startIndex,
                               int endIndex, 
                               int paths,
                               int step_for_offset_in,
                               int step_for_offset_out, 
                               int rates, 
                               thrust::device_vector<float>& swap_rate_device  
                               )
{

    swap_rate_computation_gpu(   thrust::raw_pointer_cast(&discounts_device[0])  , 
        thrust::raw_pointer_cast(&taus_device[0])  , 
        startIndex,
        endIndex, 
        paths,
        step_for_offset_in,
        step_for_offset_out, 
        rates, 
        thrust::raw_pointer_cast(&swap_rate_device[0])  // for output 
        );
}




void spot_measure_numeraires_computation_gpu_main(   thrust::device_vector<float>& discount_factors_device,
                                                  thrust::device_vector<float>& numeraire_values_device, //output
                                                   int paths,
                                                  int n_rates
                                                  )
{
    spot_measure_numeraires_computation_gpu(   thrust::raw_pointer_cast(&discount_factors_device[0]),
        thrust::raw_pointer_cast( &numeraire_values_device[0]), //output
        paths,
        n_rates
        );
}

void spot_measure_numeraires_computation_gpu_offset_main(   thrust::device_vector<float>& discount_factors_device,
                                                                                            thrust::device_vector<float>& numeraire_values_device, //output
                                                                                            int paths,
                                                                                            int pathsForOutput,
                                                                                            int pathsOffset,
                                                                                            int n_rates
                                                                                            )
{
    spot_measure_numeraires_computation_offset_gpu(   thrust::raw_pointer_cast(&discount_factors_device[0]),
        thrust::raw_pointer_cast( &numeraire_values_device[0]), //output
        paths,
        pathsForOutput,
        pathsOffset,
        n_rates
        );
}



void forward_rate_extraction_gpu_main(   thrust::device_vector<float>&all_forwards_device, 
                                                              const std::vector<int>& forwardIndices,                          
                                                              int paths,
                                                              int steps,
                                                              int rates, 
                                                              thrust::device_vector<float>& select_forwards_device                  
                                                              )
{
    if (forwardIndices.size()!= steps)
        GenerateError("Size mismatch in forward_rate_extraction_gpu_main");
    forward_rate_extraction_gpu(     thrust::raw_pointer_cast(&all_forwards_device[0]), 
                                                            forwardIndices,                          
                                                            paths,
                                                            steps,
                                                            rates, 
                                                              thrust::raw_pointer_cast(&select_forwards_device [0])               
                                                              );

}


void forward_rate_extraction_steps_gpu_main(   thrust::device_vector<float>&all_forwards_device, 
                                                              const std::vector<int>& forwardIndices,    
															  const std::vector<int>& stepIndices,                          
                                                              int paths,
                                                              int rates, 
                                                              thrust::device_vector<float>& select_forwards_device                  
                                                              )
{
    if (forwardIndices.size()!= stepIndices.size())
        GenerateError("Size mismatch in forward_rate_extraction_steps_gpu_main");
    forward_rate_extraction_selecting_step_gpu(     thrust::raw_pointer_cast(&all_forwards_device[0]), 
                                                            forwardIndices,         
															stepIndices,
                                                            paths,
                                                            rates, 
                                                              thrust::raw_pointer_cast(&select_forwards_device [0])               
                                                              );

}



double spot_measure_numeraires_extraction_main(  thrust::device_vector<float>& all_numeraire_values_device,
                                             thrust::device_vector<float>& some_numeraire_values_device, //output
                                             int pathsInput,
                                            int pathsOutput,
                                            int offsetForOutput,
                                            int numberRates,
                                            int numberExerciseTimes, 
                                            thrust::device_vector<int>& exerciseIndices_device
                                            )
{
    return spot_measure_numeraires_extraction_gpu(  thrust::raw_pointer_cast(&all_numeraire_values_device[0]),
                                            thrust::raw_pointer_cast(&some_numeraire_values_device[0]), //output
                                             pathsInput,
                                             pathsOutput,
                                             offsetForOutput,
                                             numberRates,
                                             numberExerciseTimes, 
                                            thrust::raw_pointer_cast(&exerciseIndices_device[0])
                                            );
}
