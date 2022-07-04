//
//
//                                                          LMM Evolver main .h
//
//
// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3



#ifndef LMM_EVOLVER_MAIN_H
#define LMM_EVOLVER__MAIN_H

#include <gold/pragmas.h> 

#include <thrust/device_vector.h>
#include <vector>

/*
    routine to develop many LMM paths on the device given the log rate gaussian increments
    using the log-Euler drift approximation

    it is assumed that all device vectors are correctly sized on entry
*/
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
                                                                                           thrust::device_vector<float>& e_buffer_device, 
                                                                                            thrust::device_vector<float>& evolved_rates_device, // for output
                                                                                            thrust::device_vector<float>&  evolved_log_rates_device  // for output 
                                                                                            );

/*
    routine to develop many LMM paths on the device given the log rate gaussian increments
    using the log-PC drift approximation

       it is assumed that all device vectors are correctly sized on entry
 */

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
                                                                                            );
/*
    
    routine to develop many LMM paths on the device given the uncorrelated Gaussian increments

    main innovation is that it used only one kernel call to do it all 

    using the log-PC drift approximation

       it is assumed that all device vectors are correctly sized on entry

*/

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
                         );
/*
    
    routine to develop many LMM paths on the device given the uncorrelated Gaussian increments

    main innovation is that it used only one kernel call to do it all and computes the discount factors as well if desired

    using the log-PC drift approximation

       it is assumed that all device vectors are correctly sized on entry

*/
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
                         );

/*
does all steps and discounts
optimized for kepler architecture
*/
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
                         );

/*
do all steps and discounts
main difference is that designed to work well with fermi architecture so uses fewer
textures and more shared memory
*/

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
                              );


/*
    compute all the discount ratios for a set of paths from evolved forward rates
    the df corresponding to alive index is the denominator in the ratio
    those before alive index are ignored
    we have rates +1 dfs per step 
    
    data is outputted in form

    steps \times rates+1 \times paths


    data in inputted in form 

    steps \times rates \times paths
*/

void discount_ratios_computation_main( thrust::device_vector<float>& evolved_rates_device, 
                                      thrust::device_vector<float>& taus_device, 
                                      const std::vector<int>& aliveIndices, 
                                      thrust::device_vector<int>& alive_device, 
                                      int paths,
                                      int steps, 
                                      int rates, 
                                      thrust::device_vector<float>& discounts_device,  // for output 
                                      bool allStepsAtOnce
                                      );
/*
    compute all the co-terminal swap-rates annuities from discount ratios 
    the rate runs from aliveIndex to end 


    data is outputted in form

    steps \times paths


    data in inputted in form 

    steps \times rates+1 \times paths
*/



void coterminal_annuity_ratios_computation_gpu(  thrust::device_vector<float>& discounts_device, 
                                                                                        thrust::device_vector<float>&taus_device, 
                                                                                        const std::vector<int>& aliveIndices, 
                                                                                        int paths,
                                                                                        int steps, 
                                                                                        int rates, 
                                                                                        thrust::device_vector<float>& annuities_device  // for output 
                                                                                            );



/*
    compute all the co-terminal swap-rates  from discount ratios and annuities
    
    the rates run from aliveIndex to end 

    data is outputted in form

       steps \times paths


    data in inputted in form 

    steps \times rates+1 \times paths
     steps \times paths
*/



void coterminal_swap_rates_computation_main_gpu(  thrust::device_vector<float>& discounts_device, 
                                                                                        thrust::device_vector<float>& annuities_device,
                                                                                        const std::vector<int>& aliveIndices, 
                                                                                        int paths,
                                                                                        int steps, 
                                                                                        int rates, 
                                                                                        thrust::device_vector<float>& cot_swap_rates_device  // for output 
                                                                                            );
/*
    computes swap-rate from discount ratios 

    the rate runs from startIndex to endIndex

    it only does one step -- step_for_offset_in specifies position in input data
                                            step_for_offset_out -- specifies position in input data


    You would need to call this once for each step to fully populate the output data. 

    
    data is outputted in form

    steps \times paths


    data in inputted in form 

    steps \times rates+1 \times paths
*/

void swap_rate_computation_gpu( thrust::device_vector<float>& discounts_device, 
                                                                                        thrust::device_vector<float>&taus_device, 
                                                                                        int startIndex,
                                                                                        int endIndex, 
                                                                                        int paths,
                                                                                        int step_for_offset_in,
                                                                                        int step_for_offset_out, 
                                                                                        int rates, 
                                                                                         thrust::device_vector<float>& swap_rate_device  
                                                                                            );


void spot_measure_numeraires_computation_gpu_main(   thrust::device_vector<float>& discount_factors_device,
                                                                                            thrust::device_vector<float>& numeraire_values_device, //output
                                                                                            int paths,
                                                                                            int n_rates
                                                                                            );
void spot_measure_numeraires_computation_gpu_offset_main(   thrust::device_vector<float>& discount_factors_device,
                                                                                            thrust::device_vector<float>& numeraire_values_device, //output
                                                                                            int paths,
                                                                                            int pathsForOutput,
                                                                                            int pathsOffset,
                                                                                            int n_rates
                                                                                            );

/*
    extracts specific forward rate from all forward rates, one per step


    data is outputted in form
    steps \times paths


    data in inputted in form 
    steps \times rates \times paths
*/
void forward_rate_extraction_gpu_main(   thrust::device_vector<float>&all_forwards_device, 
                                                              const std::vector<int>& forwardIndices,                          
                                                              int paths,
															  int steps,
                                                              int rates, 
                                                              thrust::device_vector<float>& select_forwards_device                  
                                                              );
// similar but you specify which steps

void forward_rate_extraction_steps_gpu_main(   thrust::device_vector<float>&all_forwards_device, 
                                                              const std::vector<int>& forwardIndices,    
															  const std::vector<int>& stepIndices,                          
                                                              int paths,
                                                              int rates, 
                                                              thrust::device_vector<float>& select_forwards_device                  
                                                              );

/*
takes the subset of numeraires you actually need for an early exercise development
*/
double spot_measure_numeraires_extraction_main(  thrust::device_vector<float>& all_numeraire_values_device,
                                             thrust::device_vector<float>& some_numeraire_values_device, //output
                                            int pathsInput,
                                            int pathsOutput,
                                            int offsetForOutput,
                                            int numberRates,
                                            int numberExerciseTimes, 
                                            thrust::device_vector<int>& exerciseIndices_device
                                            );

#endif
