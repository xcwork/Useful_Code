// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3

#ifndef CORRELATE_DRIFT_GPU_H
#define CORRELATE_DRIFT_GPU_H





extern "C"
void correlated_drift_paths_GPU(    float* input_data_device, // randon numbers
                                                                                        float* output_data_device, // correlated rate increments 
                                                                                        float* A_device, 
                                                                                        int* alive_device,
                                                                                        float* drift_fixed_device, 
                                                                                                 int A_offset_per_step,
                                                                                                 int factors, 
                                                                                                 int out_dimensions,
                                                                                                 int number_paths,
                                                                                                 int steps);
#endif
