
//
//
//                                                                                                                                         MultiD_Path_Gen_BS_Test.h
//
//
// (c) Mark Joshi 2010,2014
// This code is released under the GNU public licence version 3

/*
Purpose of this file is to contain routines to test correlation Monte Carlo code with CPU versus GPU.
*/


#ifndef MULTID_PATH_GEN_BS_TEST_H
#define MULTID_PATH_GEN_BS_TEST_H
#include "DeviceChooser.h"

int MultiDBSTestRoutine(bool verbose, bool useTextures, DeviceChooser& chooser);


#endif


