// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3



/*
            weighted_average_gpu.h

weighted basket example

*/
#ifndef WEIGHTED_AVERAGE_GPU_H
#define WEIGHTED_AVERAGE_GPU_H

extern"C"
void  basketWeightings_gpu(float* inputPaths, float* outputPaths, float* weights,
                                           int paths,
                                           int numberOfStocks,
                                           int numberSteps);

#endif
