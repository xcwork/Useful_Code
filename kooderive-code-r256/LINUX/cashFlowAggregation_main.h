
//
//                    cashFlowAggregation_main.h
//
//
//

// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3




#ifndef CASH_FLOW_AGGREGATION_MAIN_H
#define CASH_FLOW_AGGREGATION_MAIN_H
#include <thrust/device_vector.h>

void AggregateFlows_main(
                         thrust::device_vector<float>& aggregatedFlows,// output added to not overwritten
                         int totalPaths,
                         int numberExerciseDates, 
                         thrust::device_vector<float>& genFlows,
                         int batchPaths, 
                         int offSetForOutput,
                         int numberSteps, 
                         thrust::device_vector<int>& precedingIndex, // to be textured 
						 int start // first index to do
                         );
#endif
