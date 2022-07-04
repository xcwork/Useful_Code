
//
//
//                                                                                                                                          Correlation_Test.h
//
//
// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3

/*
Purpose of this file is to contain routines to test correlation Monte Carlo code with CPU versus GPU.
*/


#ifndef CORRELATION_TEST_H
#define CORRELATION_TEST_H

#include "DeviceChooser.h"


int CorrelationTestRoutine(bool verbose, bool useTextures, DeviceChooser& chooser);


#endif




