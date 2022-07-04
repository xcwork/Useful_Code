

//
//
//                          LS_Test.h
//
//
// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3

/*
Purpose of this file is to contain routines to test the least square CPU code against the least square GPU code
*/


#ifndef LS_TEST_H
#define LS_TEST_H

int Test_LS_Code(bool verbose, bool useLogBasis, bool normalise, bool useCrossTerms,bool useCula);

int Test_Basis_Variable_Extraction_etc(bool verbose);

#endif

