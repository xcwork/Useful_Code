// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3


//   AsianMain.h


#ifndef ASIAN_MAIN_H
#define ASIAN_MAIN_H

#include <gold/pragmas.h> 
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>



int AsianMainDevice(thrust::device_vector<float>& input_normals,
                                                                                                 int normalsDimensions, // must be greater than or equal to stepsPerPath
                                                                                                 int totalPaths, 
                                                                                                 int stepsPerPath,
                                                                                                 const std::vector<float>& logDrifts_vec, 
                                                                                                 const std::vector<float>& logSds_vec, 
                                                                                                 float logSpot0,
                                                                                                 float df,
                                                                                                 float strikeArithmetic,
                                                                                                 thrust::device_vector<float>& outputDataArithmetic,
                                                                                                 float strikeGeometric,
                                                                                                 thrust::device_vector<float>&outputDataGeometric);

                                                                                                 
#endif
