//
//
//                                                          LMM Evolver  GPU.h
//
//
// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3

#ifndef LMM_EVOLVER_GPU_H
#define LMM_EVOLVER_GPU_H


#include <vector>

extern "C"
void LMM_evolver_pc_gpu(  float* initial_rates_device, 
                                                                                            float* initial_log_rates_device, 
                                                                                            float* taus_device, 
                                                                                            float* correlatedBrownianIncrements_device,
                                                                                            float* pseudo_roots_device,
                                                                                            float* initial_drifts_device, 
                                                                                            float* displacements_device,
                                                                                            const std::vector<int>& aliveIndices, 
                                                                                            int paths,
                                                                                            int factors,
                                                                                            int steps, 
                                                                                            int rates, 
                                                                                            float* e_buffer_device,
                                                                                            float* e_buffer_pred_device,
                                                                                            float* evolved_rates_device, // for output
                                                                                            float* evolved_log_rates_device,  // for output 
                                                                                            bool useShared=false);

extern "C"
void LMM_evolver_euler_gpu(  float* initial_rates_device, 
                                                                                            float* initial_log_rates_device, 
                                                                                            float* taus_device, 
                                                                                            float* correlatedBrownianIncrements_device,
                                                                                            float* pseudo_roots_device,
                                                                                            float* initial_drifts_device, 
                                                                                            float* displacements_device,
                                                                                            const std::vector<int>& aliveIndices, 
                                                                                            int paths,
                                                                                            int factors,
                                                                                            int steps, 
                                                                                            int rates, 
                                                                                            float* e_buffer_device,
                                                                                            float* evolved_rates_device, // for output
                                                                                            float* evolved_log_rates_device  // for output 
                                                                                            );

 
 
extern "C"
void discount_ratios_computation_gpu(  float* evolved_rates_device, 
                                                                                        float* taus_device, 
                                                                                        const std::vector<int>& aliveIndices, 
                                                                                        int* alive_device,
                                                                                        int paths,
                                                                                        int steps, 
                                                                                        int rates, 
                                                                                        float* discounts_device,  // for output
                                                                                        bool allStepsAtOnce 
                                                                                            );

 
extern "C"
void coterminal_annuity_ratios_computation_gpu(  float* discounts_device, 
                                                                                        float* taus_device, 
                                                                                        const std::vector<int>& aliveIndices, 
                                                                                        int paths,
                                                                                        int steps, 
                                                                                        int rates, 
                                                                                        float* annuities_device  // for output 
                                                                                            );

extern "C"
void coterminal_swap_rate_computation_gpu(  float* discounts_device, 
                                                                                  float* annuities_device ,
                                                                                        const std::vector<int>& aliveIndices, 
                                                                                        int paths,
                                                                                        int steps, 
                                                                                        int rates, 
                                                                                     float* cot_swap_rates_device
                                                                                            );
 
extern "C"
void swap_rate_computation_gpu(  float* discounts_device, 
                                                                                        float* taus_device, 
                                                                                        int startIndex,
                                                                                        int endIndex, 
                                                                                        int paths,
                                                                                        int step_for_offset_in,
                                                                                        int step_for_offset_out, 
                                                                                        int rates, 
                                                                                        float* swap_rate_device  // for output 
                                                                                            );

extern "C"
void spot_measure_numeraires_computation_gpu(        float* discount_factors_global,
                                                                                                        float* numeraire_values_global, //output
                                                                                                        int paths,
                                                                                                        int n_rates);
extern "C"
void spot_measure_numeraires_computation_offset_gpu(        float* discount_factors_global,
                                             float* numeraire_values_global, //output
                                             int paths,
                                             int pathsForOutput,
                                             int pathsOffsetForOutput,
                                             int n_rates
                                             );

 
extern "C"
void forward_rate_extraction_gpu(   float* all_forwards_device, 
                                                              const std::vector<int>& forwardIndices,                          
                                                              int paths,
                                                              int steps,
                                                              int rates, 
                                                              float* select_forwards_device                  
                                                              );

extern "C"
void forward_rate_extraction_selecting_step_gpu(  float* all_forwards_device, 
                                 const std::vector<int>& forwardIndices,        
								 const std::vector<int>& stepIndices,                          
                                 int paths,
                                 int rates, 
                                 float* select_forwards_device                  
                                 );
extern "C"
double spot_measure_numeraires_extraction_gpu(   float* all_numeraire_values_global,
                                             float* some_numeraire_values_global, // output
                                            int pathsInput,
                                            int pathsOutput,
                                            int offsetForOutput,
                                            int numberRates,
                                            int numberExerciseTimes, 
                                            int* exerciseIndices_vec
                                            );

#endif
