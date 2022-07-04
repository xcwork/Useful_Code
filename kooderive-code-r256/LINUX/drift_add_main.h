
#ifndef DRIFT_ADD_MAIN_H
#define DRIFT_ADD_MAIN_H
#include <gold/pragmas.h> 
#include <thrust/device_vector.h>

void drift_add_device(thrust::device_vector<float>&device_fixed_matrix,  // matrix to add
                                                                                                 thrust::device_vector<float>& device_data, // data to add to 
                                                                                                 int steps, 
                                                                                                 int number_rates, 
                                                                                                 int paths,
                                                                                                 bool useTexture);
#endif
