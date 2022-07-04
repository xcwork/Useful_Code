// (c) Mark Joshi 2011
// This code is released under the GNU public licence version 3


//   basisFunctions.h

#ifndef BASIS_FUNCTIONS_H
#define BASIS_FUNCTIONS_H


/*
All data passed in via pointers to data already in global memory on the device.

implicit assumption that the evolution times are the rate reset dates

This routine only determines when the exercise date is. 

This is an integer in the closed range [0,numberSteps]

A lot of auxiliary data will need to be placed in LSAIntData_dev_ptr and LSAFloatData_dev_ptr,
before calling.


*/

extern "C"
void exerciseTimeDeterminationUsingLSAQuadraticSimpleBasis(int* stoppingTimes_dev_ptr, 
							   float* rebates_dev_ptr, 
							   int* LSAIntData_dev_ptr,
							   float* LSAFloatData_dev_ptr,
							   int paths, 
							   int numberSteps,                                  
							   float* rates1_dev_ptr, 
							   float* rates2_dev_ptr, 
							   float* rates3_dev_ptr, 
							   float* forwards_dev_ptr, 
							   float* discRatios_dev_ptr);
#endif //BASIS_FUNCTIONS_H

