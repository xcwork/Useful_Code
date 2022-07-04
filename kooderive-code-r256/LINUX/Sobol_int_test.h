//
//
//                                                                                                                                         Sobol_Int_Test.h
//
//
// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3

/*
Purpose of this file is to contain routines to test Sobol Monte Carlo code with CPU versus GPU.
*/


#ifndef SOBOL_INT_TEST_H
#define SOBOL_INT_TEST_H

#include "DeviceChooser.h"

int SobolTestRoutineForInts(bool verbose,DeviceChooser& chooser);



#endif
