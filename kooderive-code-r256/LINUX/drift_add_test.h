
//
//
//                                                                                                                                          Drift_add_Test.h
//
//
// (c) Mark Joshi 2009,2014
// This code is released under the GNU public licence version 3

/*
Purpose of this file is to contain routines to test correlation Monte Carlo code with CPU versus GPU.
*/


#ifndef DRIFT_ADD_TEST_H
#define DRIFT_ADD_TEST_H

#include "DeviceChooser.h"

int DriftAddTestRoutine(bool verbose, bool useTexture, DeviceChooser& chooser);


#endif

