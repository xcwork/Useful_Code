// (c) Mark Joshi 2009
// This code is released under the GNU public licence version 3
// note derived from code provided by NVIDIA 

// NVIDIA licensing terms: 
//Source Code: Developer shall have the right to modify and create derivative works with the Source Code.
//Developer shall own any derivative works ("Derivatives") it creates to the Source Code, provided that
//Developer uses the Materials in accordance with the terms and conditions of this Agreement. Developer
//may distribute the Derivatives, provided that all NVIDIA copyright notices and trademarks are used properly
//and the Derivatives include the following statement: "This software contains source code provided by
//NVIDIA Corporation."
#include <stdio.h>


#  define CUT_CHECK_ERR(errorMessage) do {                                 \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        printf("Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    err = cudaThreadSynchronize();                                           \
    if( cudaSuccess != err) {                                                \
        printf("Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    } } while (0)


#define COPYCONSTANTMEMORYFLOAT(constantPointer,sourceData) \
 CUDA_SAFE_CALL(cudaMemcpyToSymbol(constantPointer, \
                                                                                                  &sourceData[0], \
                                                                                                  sourceData.size() * sizeof(float), \
                                                                                                  0, cudaMemcpyHostToDevice));

#define COPYFROMCONSTANTMEMORYTOFLOAT(constantPointer,targetData) \
 CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&targetData[0], \
                                                                                                constantPointer, \
                                                                                                  targetData.size() * sizeof(float), \
                                                                                                  0, cudaMemcpyDeviceToHost));


#define COPYCONSTANTMEMORYINT(constantPointer,sourceData) \
 CUDA_SAFE_CALL(cudaMemcpyToSymbol(constantPointer, \
                                                                                                  &sourceData[0], \
                                                                                                  sourceData.size() * sizeof(int), \
                                                                                                  0, cudaMemcpyHostToDevice));
