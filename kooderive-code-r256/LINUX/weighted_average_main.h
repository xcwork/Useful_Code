
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3



/*
weighted basket example

*/

#ifndef WEIGHTED_AVERAGE_MAIN_H
#define WEIGHTED_AVERAGE_MAIN_H

#include <thrust/device_vector.h>

void  basketWeightings_main(thrust::device_vector<float>& inputPaths, 
                            thrust::device_vector<float>& outputPaths, 
                            thrust::device_vector<float>& weights,
                                           int paths,
                                           int numberOfStocks,
                                           int numberSteps);

#endif


