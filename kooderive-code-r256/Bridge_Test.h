//
//
//                                                                                                                                          Bridge_Test.h
//
//
// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3

/*
Purpose of this file is to contain routines to test Brownian bridge Monte Carlo code with CPU versus GPU.
*/


#ifndef BRIDGE_TEST_H
#define BRIDGE_TEST_H
#include "DeviceChooser.h"

int BridgeTestRoutine(bool verbose, bool doNormalInSobol, DeviceChooser& chooser);

int MultiDBridgeTestOrderingRoutine(bool verbose, bool useTextures, DeviceChooser& chooser);

int MultiDBridgeTestRoutine(bool verbose, bool doNormalInSobol, bool useTextures, DeviceChooser& chooser);

int Brownian_bridge_test_routine(bool verbose, bool justGPU, DeviceChooser& chooser);

#endif
