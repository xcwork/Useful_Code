
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3



/*
weighted basket example

*/

#include <weighted_average_main.h>
#include <weighted_average_gpu.h>
#include <thrust/device_vector.h>

void  basketWeightings_main(thrust::device_vector<float>& inputPaths, 
                            thrust::device_vector<float>& outputPaths, 
                            thrust::device_vector<float>& weights,
                                           int paths,
                                           int numberOfStocks,
                                           int numberSteps)
{
    basketWeightings_gpu( thrust::raw_pointer_cast(&inputPaths[0]),
         thrust::raw_pointer_cast(&outputPaths[0]), 
         thrust::raw_pointer_cast(&weights[0]),
        paths,
        numberOfStocks,
        numberSteps);
}
