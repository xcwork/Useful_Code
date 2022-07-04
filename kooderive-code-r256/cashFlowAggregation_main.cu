
//
//                    cashFlowAggregation_main.cu
//
//
//

// (c) Mark Joshi 2011. 2013
// This code is released under the GNU public licence version 3


#include <cashFlowAggregation_main.h>
#include <cashFlowAggregation_gpu.h>


void AggregateFlows_main(
                         thrust::device_vector<float>& aggregatedFlows,// output added to not overwritten
                         int totalPaths,
                         int numberExerciseDates, 
                         thrust::device_vector<float>& genFlows,
                         int batchPaths, 
                         int offSetForOutput,
                         int numberSteps, 
                         thrust::device_vector<int>& precedingIndex, // to be textured 
						 int start
                         )
{
    AggregateFlows_gpu(
                         thrust::raw_pointer_cast(& aggregatedFlows[0]),// output added to not overwritten
                          totalPaths,
                          numberExerciseDates, 
                         thrust::raw_pointer_cast(& genFlows[0]),
                         batchPaths, 
                         offSetForOutput,
                         numberSteps, 
                         thrust::raw_pointer_cast(& precedingIndex[0]), // to be textured 
                         start
						 );
}
