//
//
//       LSmulti_Test.h
//
//
// (c) Mark Joshi 2011,2012,2013,2014
// This code is released under the GNU public licence version 3

// routine to test the multi LS Code

#ifndef LSMULTI_TEST_H
#define LSMULTI_TEST_H
#include "DeviceChooser.h"

int Test_LS__Multi_Code(bool verbose, bool useLogBasis,bool useCrossTerms,  bool globallyNormalise,DeviceChooser& chooser);

int Test_Multi_LS_etc(bool verbose,DeviceChooser& chooser);
#endif
