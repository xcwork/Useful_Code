

//
//
//                                                                                                                                         LMM_evolver_Euler_Test.h
//
//
// (c) Mark Joshi 2010,2014
// This code is released under the GNU public licence version 3

/*
Purpose of this file is to contain routines to test LMM Euler Evolution Monte Carlo code with CPU versus GPU.
*/


#ifndef LMM_EVOLVER_EULER_TEST_H
#define LMM_EVOLVER_EULER_TEST_H
#include "DeviceChooser.h"

int LMMLogEulerTestRoutine(bool verbose, bool useTextures, DeviceChooser& chooser);

int LMMPCTestRoutine(bool verbose,  bool useTextures, DeviceChooser& chooser);


int LMMPCSKTestRoutine(bool verbose, int firstj, int lastj, int firstThreads, int lastThread, int threadstep, DeviceChooser& chooser);
#endif

