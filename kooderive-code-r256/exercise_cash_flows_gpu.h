
//
//
//                 exercise_cash_flows_gpu.h
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3


#ifndef EXERCISE_CASH_FLOWS_GPU_H
#define EXERCISE_CASH_FLOWS_GPU_H


extern"C"
void exercise_cash_flows_gpu(float* genFlows,  // gen flows will be modified
										   int* firstIndex, // will be changed
										   int* secondIndex, // will be changed
										   float* theta,  // will be changed
										   float* exerciseValues,
										   int* firstIndexExercise,
										   int* secondIndexExercise,
										   float* thetaExercise,
										   int* exerciseTimes, 
										   int paths, 
										   int numberSteps) ;
#endif
