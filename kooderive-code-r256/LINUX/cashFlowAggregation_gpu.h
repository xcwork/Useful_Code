
//
//                    cashFlowAggregation_gpu.h
//
//
//

// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3




#ifndef CASH_FLOW_AGGREGATION_GPU_H
#define CASH_FLOW_AGGREGATION_GPU_H

extern"C"
void AggregateFlows_gpu(
                            float* aggregatedFlows,// output added to not overwritten
                                      int totalPaths,
                                      int numberExerciseDates, 
                                      float* genFlows,
                                      int batchPaths, 
                                      int offSetForOutput,
                                      int numberSteps, 
                                      int* precedingIndex, // to be textured 
                                      int start);
#endif
